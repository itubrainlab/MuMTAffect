#!/usr/bin/env python
"""
Grid search over experimental settings:
  - Modality combinations: full, remove eye, remove pupil, remove AUs, remove GSR.
  - Branch options: 'emotion_only', 'emotion_personality', 'emotion_personality_gender'
  - Use or ignore stim_emo.

For each combination the model is trained (via multi–phase training),
evaluated on the test set, and the model is saved.
Gender loss (and F1) is computed and printed when the gender branch is activated.
"""

import argparse, os, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, math
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, r2_score


# Set default dtype
torch.set_default_dtype(torch.float32)

# Import your model and dataset definitions.
# (Assume your model file "model.py" contains AdvancedConcatFusionMultiModalModelTransformerAttention,
#  EpsilonInsensitiveLoss, compute_emotion_loss, and scale_with_sigmoid)
from model import (AdvancedConcatFusionMultiModalModelTransformerAttention_v2, 
                   EpsilonInsensitiveLoss, compute_emotion_loss, scale_with_sigmoid)
from dataset import (MultiModalDataset, common_flag_categories, gaze_cols, pupil_cols,
                     process_modal_data, desired_au_cols, desired_gsr_cols, process_stim_emo,
                     flatten_dict_values)

##########################################
# Grid search definitions
##########################################
# Modality configuration: a dict indicating whether each modality is used.
modality_options = [
    {"eye": True, "pupil": True, "au": True, "gsr": True},
    # {"eye": False, "pupil": True, "au": True, "gsr": True},
    # {"eye": True, "pupil": False, "au": True, "gsr": True},
    # {"eye": True, "pupil": True, "au": False, "gsr": True},
    # {"eye": True, "pupil": True, "au": True, "gsr": False},
]

# Branch options: controls which outputs are trained.
branch_options = ["emotion_personality","emotion_only"]

# Use stim_emo flag: whether to include stim_emo input.
stim_emo_options = [True, False]

##########################################
# (Assumed) Training phase functions from your original script.
# They include: run_phase1, run_phase1_5_emotion_warmup, run_phase2, run_phase3, run_phase4.
# For brevity we assume they are already defined as in your code.
##########################################

##########################################
# Helper: Save model checkpoint
##########################################
def save_model(model, model_name, version="finetuned"):
    current_date = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join("models", f"{model_name}_{current_date}")
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{model_name}_{version}.pth"
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return save_path

##########################################
# Helper: Modify batch based on modality config
##########################################
def apply_modality_mask(batch, modality_config):
    # batch: tuple (eye, pupil, au, gsr, stim_emo, personality, emotion, user_id, eye_features, au_features, shimmer_features, gender)
    # For each modality, if modality_config[modality] is False, replace the tensor with zeros.
    eye, pupil, au, gsr, stim_emo, personality, emotion, user_id, eye_features, au_features, shimmer_features, gender = batch
    if not modality_config.get("eye", True):
        eye = torch.zeros_like(eye)
    if not modality_config.get("pupil", True):
        pupil = torch.zeros_like(pupil)
    if not modality_config.get("au", True):
        au = torch.zeros_like(au)
    if not modality_config.get("gsr", True):
        gsr = torch.zeros_like(gsr)
    return (eye, pupil, au, gsr, stim_emo, personality, emotion, user_id, eye_features, au_features, shimmer_features, gender)


##################################################
# Class Weights for Emotions
##################################################
def compute_emotion_class_weights(df, emotion_columns):
    """
    Convert labels 1–9 => 0–8 => 3 classes => Balanced weights for CE.
    """
    weights = {}
    num_classes = 3
    for emo in emotion_columns:
        labels = df[emo].values.astype(np.int64) - 1
        binned = labels // 3
        counts = np.bincount(binned, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0  # avoid zero-division
        inv_counts = 1.0 / counts
        norm_weights = inv_counts * (num_classes / inv_counts.sum())
        weights[emo] = torch.tensor(norm_weights)
        print(f"{emo}: counts={counts}, weights={norm_weights}")
    return weights

##############################################
# Phase training functions
##############################################
# def print_tensor_stats(name, tensor):
#     print(f"{name}: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def print_tensor_stats(name, tensor):
    if tensor is None:
        print(f"{name}: None")
    else:
        try:
            print(f"{name}: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, "
                  f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
        except Exception as e:
            print(f"{name}: Could not compute stats: {e}")

def run_phase1(model, train_loader, val_loader, optimizer, scheduler, num_epochs, r2_threshold, patience,test=False):
    # Freeze emotion modules.
    for param in model.emotion_heads.parameters():
        param.requires_grad = False
    for param in model.hierarchical_conv_emotion.parameters():
        param.requires_grad = False
    for param in model.fuse_fc_emo.parameters():
        param.requires_grad = False

    # If the model has a gender branch, define a gender loss.
    if hasattr(model, "gender_branch") and (model.gender_branch is not None):
        criterion_gender = nn.CrossEntropyLoss().to(device)
        lambda_gender = 1.0  # weight for gender loss (adjust as needed)
    else:
        criterion_gender = None

    best_r2 = -1e6
    patience_counter = 0
    if test:
        num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            # Unpack batch: note that the dataset now returns an extra gender tensor.
            eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad()
            trial_delta, raw_person, _, _, gender_pred = model(
                eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                eye_features, au_features, shimmer_features
            )
            loss_person = criterion_personality_1(raw_person, personality_target)
            if criterion_gender is not None:
                loss_gender = criterion_gender(gender_pred, gender)
                loss = loss_person + lambda_gender * loss_gender
            else:
                loss = loss_person
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        gender_preds_list, gender_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                    x.to(device) for x in batch
                ]
                trial_delta, raw_person, _, _, gender_pred = model(
                    eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                    eye_features, au_features, shimmer_features
                )
                loss_person = criterion_personality_1(raw_person, personality_target)
                if criterion_gender is not None:
                    loss_gender = criterion_gender(gender_pred, gender)
                    loss = loss_person + lambda_gender * loss_gender
                else:
                    loss = loss_person
                val_loss += loss.item()
                all_preds.append(raw_person.cpu().numpy())
                all_targets.append(batch[5].cpu().numpy())
                if criterion_gender is not None:
                    # For gender, get predicted class via argmax.
                    gender_out = torch.argmax(gender_pred, dim=1)
                    gender_preds_list.append(gender_out.cpu().numpy())
                    gender_targets_list.append(gender.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        try:
            r2_scores = r2_score(targets, preds, multioutput="raw_values")
            r2 = np.mean(r2_scores)
        except Exception as e:
            print("Phase 1 r2 calculation error:", e)
            raise optuna.exceptions.TrialPruned("Phase 1 r2 calculation error: " + str(e))
        if np.isnan(r2):
            raise optuna.exceptions.TrialPruned("Phase 1 r2 is NaN.")
        scheduler.step()

        print(f"[Phase 1] Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val R²: {r2:.4f}")
        if criterion_gender is not None:
            gender_preds = np.concatenate(gender_preds_list, axis=0)
            gender_targets = np.concatenate(gender_targets_list, axis=0)
            gender_f1 = f1_score(gender_targets, gender_preds, average="macro", zero_division=0)
            print(f"          Val Gender F1: {gender_f1:.4f}")
        if (epoch + 1) % 5 == 0:
            print(f"--- Validation at Epoch {epoch+1} ---")
            print("R² for Personality:", r2_scores)
            print("-------------------------------\n")
        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0
        else:
            patience_counter += 1
        if r2 >= r2_threshold or patience_counter >= patience:
            print(f"Phase 1 stopping at epoch {epoch+1} with R²: {r2:.4f}")
            break
    return model, best_r2


def run_phase1_5_emotion_warmup(model, train_loader, val_loader, optimizer, scheduler, num_epochs, f1_threshold, patience,test=False):
    # Freeze all except emotion modules.
    for param in model.emotion_heads.parameters():
        param.requires_grad = True
    for param in model.hierarchical_conv_emotion.parameters():
        param.requires_grad = True
    for param in model.fuse_fc_emo.parameters():
        param.requires_grad = True

    best_f1 = -float("inf")
    patience_counter = 0
    alpha = 0.3
    if test:
        num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_emo_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad()
            _, raw_person, emotion_logits, _, _ = model(
                eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                eye_features, au_features, shimmer_features
            )
            loss_person = criterion_personality_1(raw_person, personality_target)
            loss_emo = compute_emotion_loss(emotion_logits, emotion_target, emotion_class_weights, gamma=2.0)
            loss = alpha * loss_person + (1 - alpha) * loss_emo
            loss.backward()
            optimizer.step()
            running_emo_loss += loss_emo.item()
        avg_train_emo_loss = running_emo_loss / len(train_loader)

        model.eval()
        val_emo_loss = 0.0
        all_emo_preds = [[] for _ in range(emotion_target.shape[1])]
        all_emo_targets = [[] for _ in range(emotion_target.shape[1])]
        gender_preds_list, gender_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                    x.to(device) for x in batch
                ]
                _, _, emotion_logits, _, gender_pred = model(
                    eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                    eye_features, au_features, shimmer_features
                )
                loss_emo = compute_emotion_loss(emotion_logits, emotion_target, emotion_class_weights, gamma=2.0)
                val_emo_loss += loss_emo.item()
                for i in range(emotion_target.shape[1]):
                    preds_i = torch.argmax(emotion_logits[:, i, :], dim=1)
                    all_emo_preds[i].append(preds_i.cpu().numpy())
                    all_emo_targets[i].append(batch[6][:, i].numpy())
                if hasattr(model, "gender_branch") and model.gender_branch is not None:
                    gender_out = torch.argmax(gender_pred, dim=1)
                    gender_preds_list.append(gender_out.cpu().numpy())
                    gender_targets_list.append(gender.cpu().numpy())
        avg_val_emo_loss = val_emo_loss / len(val_loader)
        f1_list = []
        for i in range(emotion_target.shape[1]):
            preds = np.concatenate(all_emo_preds[i])
            targets = np.concatenate(all_emo_targets[i])
            f1_val = f1_score(targets, preds, average="macro", zero_division=0)
            f1_list.append(f1_val)
        avg_f1 = np.mean(f1_list)
        if hasattr(model, "gender_branch") and model.gender_branch is not None:
            gender_preds = np.concatenate(gender_preds_list)
            gender_targets = np.concatenate(gender_targets_list)
            gender_f1 = f1_score(gender_targets, gender_preds, average="macro", zero_division=0)
        else:
            gender_f1 = None
        scheduler.step()

        print(f"[Phase 1.5 - Warmup] Epoch {epoch+1}/{num_epochs} => Train Emo Loss {avg_train_emo_loss:.4f}, Val Emo Loss {avg_val_emo_loss:.4f}, Val Emotion F1: {avg_f1:.4f}")
        if gender_f1 is not None:
            print(f"          Val Gender F1: {gender_f1:.4f}")
        if (epoch + 1) % 5 == 0:
            print("Avg. Emotion Macro F1:", avg_f1)
            for i, emo in enumerate(selected_emotions):
                print(f"{emo} => Macro F1: {f1_list[i]:.4f}")
            print("-------------------------------\n")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience_counter = 0
        else:
            patience_counter += 1
        if best_f1 >= f1_threshold or patience_counter >= patience:
            print(f"Early stopping warmup at epoch {epoch+1} with F1: {avg_f1:.4f}")
            break

    return model, best_f1

def run_phase2(model, train_loader, val_loader, optimizer, scheduler, num_epochs, f1_threshold, patience, alpha=0.2, use_personality=True,test=False):
    # Unfreeze emotion modules.
    for param in model.emotion_heads.parameters():
        param.requires_grad = True
    for param in model.hierarchical_conv_emotion.parameters():
        param.requires_grad = True
    for param in model.fuse_fc_emo.parameters():
        param.requires_grad = True

    # If model has a gender branch, define a gender loss.
    if hasattr(model, "gender_branch") and (model.gender_branch is not None):
        criterion_gender = nn.CrossEntropyLoss().to(device)
        lambda_gender = 1.0  # adjust as needed
    else:
        criterion_gender = None

    best_metric = float('inf')
    best_metric_2=0
    patience_counter = 0
    if test:
        num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_person = 0.0
        running_emo = 0.0
        running_gender = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            # Unpack batch.
            eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad()
            trial_delta, raw_person, emotion_logits, _, gender_pred = model(
                eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                eye_features, au_features, shimmer_features
            )
            # Compute personality loss only if use_personality is True.
            loss_person = criterion_personality_2(raw_person, personality_target) if use_personality else 0.0
            loss_emo = compute_emotion_loss(emotion_logits, emotion_target, emotion_class_weights, gamma=2.0)
            if criterion_gender is not None:
                loss_gender = criterion_gender(gender_pred, gender)
                loss = alpha * loss_person + (1 - alpha) * loss_emo + lambda_gender * loss_gender
                running_gender += loss_gender.item()
            else:
                loss = alpha * loss_person + (1 - alpha) * loss_emo
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_person += loss_person if isinstance(loss_person, float) else loss_person.item()
            running_emo += loss_emo.item()
        avg_train_person = running_person / len(train_loader)
        avg_train_emo = running_emo / len(train_loader)
        if criterion_gender is not None:
            avg_train_gender = running_gender / len(train_loader)
        else:
            avg_train_gender = None

        model.eval()
        val_person_loss = 0.0
        val_emo_loss = 0.0
        val_gender_loss = 0.0
        all_preds, all_targets = [], []
        # For emotion:
        all_emo_preds = [[] for _ in range(emotion_target.shape[1])]
        all_emo_targets = [[] for _ in range(emotion_target.shape[1])]
        # For gender:
        gender_preds_list, gender_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                    x.to(device) for x in batch
                ]
                trial_delta, raw_person, emotion_logits, _, gender_pred = model(
                    eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                    eye_features, au_features, shimmer_features
                )
                loss_person = criterion_personality_2(raw_person, personality_target) if use_personality else 0.0
                loss_emo = compute_emotion_loss(emotion_logits, emotion_target, emotion_class_weights, gamma=2.0)
                if criterion_gender is not None:
                    loss_gender = criterion_gender(gender_pred, gender)
                    loss = alpha * loss_person + (1 - alpha) * loss_emo + lambda_gender * loss_gender
                    val_gender_loss += loss_gender.item()
                    gender_out = torch.argmax(gender_pred, dim=1)
                    gender_preds_list.append(gender_out.cpu().numpy())
                    gender_targets_list.append(gender.cpu().numpy())
                else:
                    loss = alpha * loss_person + (1 - alpha) * loss_emo
                val_person_loss += loss_person if isinstance(loss_person, float) else loss_person.item()
                val_emo_loss += loss_emo.item()
                all_preds.append(raw_person.cpu().numpy())
                all_targets.append(batch[5].cpu().numpy())
                for i in range(emotion_target.shape[1]):
                    preds_i = torch.argmax(emotion_logits[:, i, :], dim=1)
                    all_emo_preds[i].append(preds_i.cpu().numpy())
                    all_emo_targets[i].append(batch[6][:, i].cpu().numpy())
        avg_val_person = val_person_loss / len(val_loader)
        avg_val_emo = val_emo_loss / len(val_loader)
        if criterion_gender is not None:
            avg_val_gender = val_gender_loss / len(val_loader)
        else:
            avg_val_gender = None

        val_loss = avg_val_person + avg_val_emo
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        try:
            r2_scores = r2_score(targets, preds, multioutput="raw_values")
            r2 = np.mean(r2_scores)
        except Exception as e:
            print("Phase 2 r2 calculation error:", e)
            raise optuna.exceptions.TrialPruned("Phase 2 r2 calculation error: " + str(e))
        if np.isnan(r2):
            raise optuna.exceptions.TrialPruned("Phase 2 r2 is NaN.")
        try:
            f1_list = []
            for i in range(emotion_target.shape[1]):
                p = np.concatenate(all_emo_preds[i])
                t = np.concatenate(all_emo_targets[i])
                f1_val = f1_score(t, p, average="macro", zero_division=0)
                f1_list.append(f1_val)
            avg_f1 = np.mean(f1_list)
        except Exception as e:
            print("Phase 2 f1 calculation error:", e)
            raise optuna.exceptions.TrialPruned("Phase 2 f1 calculation error: " + str(e))
        if np.isnan(avg_f1):
            raise optuna.exceptions.TrialPruned("Phase 2 f1 is NaN.")
        if criterion_gender is not None:
            gender_preds = np.concatenate(gender_preds_list)
            gender_targets = np.concatenate(gender_targets_list)
            gender_f1 = f1_score(gender_targets, gender_preds, average="macro", zero_division=0)
        else:
            gender_f1 = None

        scheduler.step()
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        print(f"[Phase 2] Epoch {epoch+1}: LR: {current_lrs}")
        print(f"           Train: Person Loss {avg_train_person:.4f}, Emo Loss {avg_train_emo:.4f}" + 
              (f", Gender Loss {avg_train_gender:.4f}" if avg_train_gender is not None else "") +
              f" | Val: Person Loss {avg_val_person:.4f}, Emo Loss {avg_val_emo:.4f}" +
              (f", Gender Loss {avg_val_gender:.4f}" if avg_val_gender is not None else "") +
              f" | Val R²: {r2:.4f}, Val Emotion F1: {avg_f1:.4f}" +
              (f", Val Gender F1: {gender_f1:.4f}" if gender_f1 is not None else ""))
        if (epoch + 1) % 5 == 0:
            print(f"--- Validation at Epoch {epoch+1} ---")
            print("R² for Personality:", r2_scores)
            print("Avg. Emotion Macro F1:", avg_f1)
            for i, emo in enumerate(selected_emotions):
                print(f"{emo} => Macro F1: {f1_list[i]:.4f}")
            if gender_f1 is not None:
                print(f"Gender Macro F1: {gender_f1:.4f}")
            print("-------------------------------\n")
        metric = avg_val_person + avg_val_emo  # you may also incorporate gender loss/metric
        metric_2 = avg_f1
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_path_phase2.pth")
        elif metric_2 > best_metric_2:
            best_metric_2 = metric_2
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_path_phase2.pth")
        else:
            patience_counter += 1
        if avg_f1 >= f1_threshold or patience_counter >= patience:
            print(f"Phase 2 stopping at epoch {epoch+1} with Val Emotion F1: {avg_f1:.4f}")
            break
    return model, r2, avg_f1


def run_phase3(model, train_loader, val_loader, optimizer, scheduler, num_epochs, r2_threshold, patience, use_personality=True,test=False):
    # Freeze all layers except personality branches.
    for param in model.parameters():
        param.requires_grad = False
    for module in [model.rawpers_fc, model.hierarchical_conv_personality, model.fuse_fc_per, model.trial_fc]:
        for param in module.parameters():
            param.requires_grad = True

    # If the model has a gender branch, define a gender loss.
    if hasattr(model, "gender_branch") and (model.gender_branch is not None):
        criterion_gender = nn.CrossEntropyLoss().to(device)
        lambda_gender = 1.0  # adjust as needed
    else:
        criterion_gender = None

    best_r2 = -1e6
    patience_counter = 0
    if test:
        num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_gender_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Phase 3 Epoch {epoch+1}", leave=False):
            eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad()
            trial_delta, raw_person, _, _, gender_pred = model(
                eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                eye_features, au_features, shimmer_features
            )
            # Only compute personality loss if use_personality is True.
            loss_person = criterion_personality_1(raw_person, personality_target) if use_personality else 0.0
            if criterion_gender is not None:
                loss_gender = criterion_gender(gender_pred, gender)
                loss = loss_person + lambda_gender * loss_gender
                running_gender_loss += loss_gender.item()
            else:
                loss = loss_person
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        if criterion_gender is not None:
            avg_train_gender_loss = running_gender_loss / len(train_loader)
        else:
            avg_train_gender_loss = None

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        gender_preds_list, gender_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                    x.to(device) for x in batch
                ]
                trial_delta, raw_person, _, _, gender_pred = model(
                    eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                    eye_features, au_features, shimmer_features
                )
                loss_person = criterion_personality_1(raw_person, personality_target) if use_personality else 0.0
                val_loss += loss_person.item()
                all_preds.append(raw_person.cpu().numpy())
                all_targets.append(batch[5].cpu().numpy())
                if criterion_gender is not None:
                    gender_out = torch.argmax(gender_pred, dim=1)
                    gender_preds_list.append(gender_out.cpu().numpy())
                    gender_targets_list.append(gender.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        try:
            r2_scores = r2_score(targets, preds, multioutput="raw_values")
            r2 = np.mean(r2_scores)
        except Exception as e:
            print("Phase 3 r2 calculation error:", e)
            raise optuna.exceptions.TrialPruned("Phase 3 r2 calculation error: " + str(e))
        if np.isnan(r2):
            raise optuna.exceptions.TrialPruned("Phase 3 r2 is NaN.")
        scheduler.step()
        print(f"[Phase 3] Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}" +
              (f", Train Gender Loss: {avg_train_gender_loss:.4f}" if avg_train_gender_loss is not None else "") +
              f" | Val Loss: {avg_val_loss:.4f} | Val R²: {r2:.4f}")
        if criterion_gender is not None:
            gender_preds = np.concatenate(gender_preds_list, axis=0)
            gender_targets = np.concatenate(gender_targets_list, axis=0)
            gender_f1 = f1_score(gender_targets, gender_preds, average="macro", zero_division=0)
            print(f"           Val Gender F1: {gender_f1:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0
        else:
            patience_counter += 1
        if r2 >= r2_threshold or patience_counter >= patience:
            print(f"Phase 3 stopping at epoch {epoch+1} with R²: {r2:.4f}")
            break
    return model, best_r2



def run_phase4(model, train_loader, val_loader, optimizer, scheduler, num_epochs,test=False):
    # Freeze all layers except emotion branches and trial_fc.
    for param in model.parameters():
        param.requires_grad = False
    for module in [model.emotion_heads, model.hierarchical_conv_emotion, model.fuse_fc_emo, model.trial_fc]:
        for param in module.parameters():
            param.requires_grad = True

    # If gender branch exists, define criterion for gender.
    if hasattr(model, "gender_branch") and (model.gender_branch is not None):
        criterion_gender = nn.CrossEntropyLoss().to(device)
    else:
        criterion_gender = None
    if test:
        num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad()
            trial_delta, raw_person, emotion_logits, _ , gender_pred = model(
                eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                eye_features, au_features, shimmer_features
            )
            loss_emo = compute_emotion_loss(emotion_logits, emotion_target, emotion_class_weights, gamma=2.0)
            if criterion_gender is not None:
                loss_gender = criterion_gender(gender_pred, gender)
                loss = loss_emo + loss_gender
            else:
                loss = loss_emo
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        print(f"[Phase 4] Epoch {epoch+1}: Train Emotion Loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        all_emo_preds = [[] for _ in range(emotion_target.shape[1])]
        all_emo_targets = [[] for _ in range(emotion_target.shape[1])]
        gender_preds_list, gender_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                    x.to(device) for x in batch
                ]
                trial_delta, raw_person, emotion_logits, _ , gender_pred = model(
                    eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                    eye_features, au_features, shimmer_features
                )
                loss_emo = compute_emotion_loss(emotion_logits, emotion_target, emotion_class_weights, gamma=2.0)
                val_loss += loss_emo.item()
                for i in range(emotion_target.shape[1]):
                    preds = torch.argmax(emotion_logits[:, i, :], dim=1)
                    all_emo_preds[i].append(preds.cpu().numpy())
                    all_emo_targets[i].append(batch[6][:, i].cpu().numpy())
                if criterion_gender is not None:
                    gender_out = torch.argmax(gender_pred, dim=1)
                    gender_preds_list.append(gender_out.cpu().numpy())
                    gender_targets_list.append(gender.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        try:
            f1_list = []
            for i in range(emotion_target.shape[1]):
                preds_temp = np.concatenate(all_emo_preds[i])
                targets_temp = np.concatenate(all_emo_targets[i])
                f1_val = f1_score(targets_temp, preds_temp, average="macro", zero_division=0)
                f1_list.append(f1_val)
            avg_f1 = np.mean(f1_list)
        except Exception as e:
            print("Phase 4 f1 calculation error:", e)
            raise optuna.exceptions.TrialPruned("Phase 4 f1 calculation error: " + str(e))
        if np.isnan(avg_f1):
            raise optuna.exceptions.TrialPruned("Phase 4 f1 is NaN.")
        if criterion_gender is not None:
            gender_preds = np.concatenate(gender_preds_list)
            gender_targets = np.concatenate(gender_targets_list)
            gender_f1 = f1_score(gender_targets, gender_preds, average="macro", zero_division=0)
        else:
            gender_f1 = None
        print(f"[Phase 4] Epoch {epoch+1}: Train Emotion Loss: {avg_train_loss:.4f}, Val Emotion Loss: {avg_val_loss:.4f}, Val Emotion Macro F1: {avg_f1:.4f}" +
              (f", Val Gender Macro F1: {gender_f1:.4f}" if gender_f1 is not None else ""))
    model.eval()
    val_loss = 0.0
    all_emo_preds = [[] for _ in range(emotion_target.shape[1])]
    all_emo_targets = [[] for _ in range(emotion_target.shape[1])]
    gender_preds_list, gender_targets_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                x.to(device) for x in batch
            ]
            trial_delta, raw_person, emotion_logits, _ , gender_pred = model(
                eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                eye_features, au_features, shimmer_features
            )
            loss_emo = compute_emotion_loss(emotion_logits, emotion_target, emotion_class_weights, gamma=2.0)
            val_loss += loss_emo.item()
            for i in range(emotion_target.shape[1]):
                preds = torch.argmax(emotion_logits[:, i, :], dim=1)
                all_emo_preds[i].append(preds.cpu().numpy())
                all_emo_targets[i].append(batch[6][:, i].cpu().numpy())
            if criterion_gender is not None:
                gender_out = torch.argmax(gender_pred, dim=1)
                gender_preds_list.append(gender_out.cpu().numpy())
                gender_targets_list.append(gender.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    try:
        f1_list = []
        for i in range(emotion_target.shape[1]):
            preds_temp = np.concatenate(all_emo_preds[i])
            targets_temp = np.concatenate(all_emo_targets[i])
            f1_val = f1_score(targets_temp, preds_temp, average="macro", zero_division=0)
            f1_list.append(f1_val)
        avg_f1 = np.mean(f1_list)
    except Exception as e:
        print("Phase 4 f1 calculation error:", e)
        raise optuna.exceptions.TrialPruned("Phase 4 f1 calculation error: " + str(e))
    if np.isnan(avg_f1):
        raise optuna.exceptions.TrialPruned("Phase 4 f1 is NaN.")
    if criterion_gender is not None:
        gender_preds = np.concatenate(gender_preds_list)
        gender_targets = np.concatenate(gender_targets_list)
        gender_f1 = f1_score(gender_targets, gender_preds, average="macro", zero_division=0)
    else:
        gender_f1 = None
    print(f"[Phase 4] Final Val Emotion Loss: {avg_val_loss:.4f}, Val Emotion Macro F1: {avg_f1:.4f}" +
          (f", Val Gender Macro F1: {gender_f1:.4f}" if gender_f1 is not None else ""))
    return model, avg_f1


# Helper to gather backbone parameters regardless of the model type.
def get_base_params(model):
    base_params = []
    for modality in ['eye', 'pupil', 'au', 'gsr']:
        # Check for LSTM, GRU, TCN, or Transformer-based modules.
        if hasattr(model, f'{modality}_lstm'):
            base_params += list(getattr(model, f'{modality}_lstm').parameters())
        elif hasattr(model, f'{modality}_gru'):
            base_params += list(getattr(model, f'{modality}_gru').parameters())
        elif hasattr(model, f'{modality}_tcn'):
            base_params += list(getattr(model, f'{modality}_tcn').parameters())
        elif hasattr(model, f'{modality}_transformer'):
            proj_name = f'{modality}_proj'
            base_params += list(getattr(model, proj_name).parameters())
            base_params += list(getattr(model, f'{modality}_transformer').parameters())
    # Add common layers.
    base_params += list(model.project_modality.parameters())
    base_params += list(model.modal_transformer.parameters())
    base_params += [model.pos_embedding]
    base_params += list(model.hierarchical_conv_emotion.parameters())
    base_params += list(model.fuse_fc_emo.parameters())
    base_params += list(model.hierarchical_conv_personality.parameters())
    base_params += list(model.fuse_fc_per.parameters())
    base_params += list(model.eye_feat_mlp.parameters())
    base_params += list(model.au_feat_mlp.parameters())
    base_params += list(model.shimmer_feat_mlp.parameters())
    base_params += list(model.features_mlps.parameters())
    base_params += list(model.fusion_mlp.parameters())
    return base_params


##########################################
# Main Grid Search Loop
##########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for multimodal model variants")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV/PKL data file")
    parser.add_argument("--batch_size", type=int, default=32)
    # Fixed model dimensions:
    parser.add_argument("--eye_dim", type=int, default=28)
    parser.add_argument("--pupil_dim", type=int, default=12)
    parser.add_argument("--au_dim", type=int, default=46)
    parser.add_argument("--gsr_dim", type=int, default=15)
    parser.add_argument("--stim_emo_dim", type=int, default=6)
    parser.add_argument("--personality_dim", type=int, default=5)
    parser.add_argument("--emotion_num_classes", type=int, default=3)
    parser.add_argument("--eye_feat_dim", type=int, default=17)
    parser.add_argument("--au_feat_dim", type=int, default=128)
    parser.add_argument("--shimmer_feat_dim", type=int, default=40)
    parser.add_argument("--embedding_noise_std", type=float, default=0.0)
    parser.add_argument("--weight_decay_phase2", type=float, default=1e-5)
    args = parser.parse_args()
    # Set hyperparameters for grid search (modify these values as needed)
    lr_base = 8e-4
    lr_person = 5e-5
    lr_emotion = 5e-4

    # Load and preprocess data.
    df = pd.read_pickle(args.data_path)
    df = df[df['run'] != 0]
    # Optionally drop other rows.
    df = df[df['ex'] != 1]
    df.drop(columns=['lux','tem'], inplace=True)
    df = df.dropna()  # or fillna, as appropriate.
    
    unique_users = df["user"].unique()
    user2idx = {user: i for i, user in enumerate(unique_users)}
    personality_cols_list = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']
    scaler_personality = StandardScaler()
    df[personality_cols_list] = scaler_personality.fit_transform(df[personality_cols_list])
    np.random.seed(43)
    test_users = np.random.choice(unique_users, size=int(0.15 * len(unique_users)), replace=False)
    print("Test users:", test_users)
    test_df = df[df['user'].isin(test_users)].copy()
    df_train = df[~df['user'].isin(test_users)].copy()
    train_df, val_df = train_test_split(df_train, test_size=0.1765, random_state=43)
    # Split data: train, validation, test.
    # train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=43)
    # train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=43)
    test_df = test_df.reset_index(drop=True)
    
   
    
    # Prepare scalers for modalities.
    eye_data_list = train_df['Eye_Data'].apply(lambda x: process_modal_data(x, gaze_cols, common_flag_categories)).tolist()
    eye_data = np.concatenate(eye_data_list, axis=0)
    scaler_eye = StandardScaler().fit(eye_data)
    
    pupil_data_list = train_df['Eye_Data'].apply(lambda x: process_modal_data(x, pupil_cols, common_flag_categories)).tolist()
    pupil_data = np.concatenate(pupil_data_list, axis=0)
    scaler_pupil = StandardScaler().fit(pupil_data)
    
    au_data_list = train_df['AUs'].apply(lambda x: process_modal_data(x, desired_au_cols, common_flag_categories)).tolist()
    au_data = np.concatenate(au_data_list, axis=0)
    scaler_au = StandardScaler().fit(au_data)
    
    shimmer_data_list = train_df['Shimmer'].apply(lambda x: process_modal_data(x, desired_gsr_cols, common_flag_categories)).tolist()
    shimmer_data = np.concatenate(shimmer_data_list, axis=0)
    scaler_shimmer = StandardScaler().fit(shimmer_data)
    
    modality_scalers = {
        'Eye_Data': scaler_eye,
        'Pupil_Data': scaler_pupil,
        'AUs': scaler_au,
        'Shimmer': scaler_shimmer
    }
    
    # Create datasets.
    # For example, here we use a subset of emotions.
    selected_emotions = ['felt_arousal','felt_valance']
    train_dataset = MultiModalDataset(train_df, selected_emotions, user2idx, modality_scalers=modality_scalers)
    val_dataset   = MultiModalDataset(val_df, selected_emotions, user2idx, modality_scalers=modality_scalers)
    test_dataset  = MultiModalDataset(test_df, selected_emotions, user2idx, modality_scalers=modality_scalers)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    scaled_train = scaler_personality.transform(train_df[personality_cols_list])
    max_val_value = np.abs(scaled_train).max()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device on Apple Silicon.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU fallback.")
    print("Using device:", device)
    
    # Compute emotion class weights.
    def compute_emotion_class_weights(df, emotion_columns):
        weights = {}
        num_classes = 3
        for emo in emotion_columns:
            labels = df[emo].values.astype(np.int64) - 1
            binned = labels // 3
            counts = np.bincount(binned, minlength=num_classes).astype(np.float32)
            counts[counts==0] = 1.0
            inv_counts = 1.0 / counts
            norm_weights = inv_counts * (num_classes / inv_counts.sum())
            weights[emo] = torch.tensor(norm_weights)
            print(f"{emo}: counts={counts}, weights={norm_weights}")
        return weights

    # # ----------------------
    # # Set deterministic algorithms and seeds.
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # import random
    # random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.autograd.set_detect_anomaly(True)
    # ----------------------
    
    # Compute emotion class weights, create datasets, etc.
    cw_dict = compute_emotion_class_weights(train_df, selected_emotions)
    emotion_class_weights = [cw_dict[emo].to(device) for emo in selected_emotions]
    
    ##########################################
    # Grid search over combinations.
    ##########################################
    test=False
    results = []
    grid_id = 0
    criterion_personality_1 = EpsilonInsensitiveLoss(epsilon=0.0, reduction='mean').to(device)  # L1 loss
    criterion_personality_2 = EpsilonInsensitiveLoss(epsilon=0.01, reduction='mean').to(device)     # L1 loss with epsilon
    
    ##########################################
    # Grid Search with 5-Fold Cross Validation and Test Set Evaluation
    ##########################################
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    results_test = []
    grid_id = 0
    for modality_config in modality_options:
        for branch_option in branch_options:
            for use_stim_emo in stim_emo_options:
                use_personality = (branch_option != "emotion_only")
                use_gender = (branch_option == "emotion_personality_gender")
                print(f"\n==== Grid {grid_id}: modalities={modality_config}, branch_option={branch_option}, use_stim_emo={use_stim_emo} ====")
                
                fold_metrics = []
                # for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
                    # print(f"\n--- Grid {grid_id} | Fold {fold+1} ---")
                    # train_fold = torch.utils.data.Subset(train_dataset, train_idx)
                    # val_fold = torch.utils.data.Subset(train_dataset, val_idx)
                    # train_loader_cv = DataLoader(train_fold, batch_size=args.batch_size, shuffle=True)
                    # val_loader_cv = DataLoader(val_fold, batch_size=args.batch_size, shuffle=False)
                    
                    # Instantiate a fresh model for this fold.
                # use_personality = True
                use_gender = (branch_option == "emotion_personality_gender")
                
                print(f"\n==== Grid {grid_id}: modalities={modality_config}, branch_option={branch_option}, use_stim_emo={use_stim_emo} ====")
                
                model = AdvancedConcatFusionMultiModalModelTransformerAttention_v2(
                    eye_dim=args.eye_dim,
                    pupil_dim=args.pupil_dim,
                    au_dim=args.au_dim,
                    gsr_dim=args.gsr_dim,
                    stim_emo_dim=(args.stim_emo_dim if use_stim_emo else 0),
                    hidden_dim=64,
                    target_length=16,
                    personality_dim=args.personality_dim,
                    emotion_num_classes=args.emotion_num_classes,
                    num_users=len(unique_users),
                    eye_feat_dim=args.eye_feat_dim,
                    au_feat_dim=args.au_feat_dim,
                    shimmer_feat_dim=args.shimmer_feat_dim,
                    proj_dim=16,
                    transformer_nhead=8,
                    transformer_layers=2,
                    dropout_person=0.5,
                    dropout_emotion=0.4,
                    dropout_features=0.3,
                    embedding_noise_std=args.embedding_noise_std,
                    max_val=max_val_value,
                    emotion_num_labels=len(selected_emotions),
                    use_gender=use_gender,
                    use_personality=use_personality,
                    use_stim_emo=use_stim_emo
                ).to(device)
                
                model.apply(lambda m: init_weights(m))
                
                # --- Training Phases for the fold ---
                if use_personality:
                    optimizer_phase1 = optim.Adam([p for p in model.parameters() if p.requires_grad],
                                            lr=0.0001, weight_decay=1e-5)
                    scheduler_phase1 = optim.lr_scheduler.ExponentialLR(optimizer_phase1, gamma=0.99)
                    model, phase1_r2 = run_phase1(model, train_loader, val_loader, optimizer_phase1, scheduler_phase1,
                                                num_epochs=50, r2_threshold=0.30, patience=5, test=test)
                    print(f"Phase 1 final personality R²: {phase1_r2:.4f}")
                    
                    # print("\n--- Phase 1.5: Emotion Warmup ---")
                    # optimizer_emotion_warmup = optim.Adam([p for p in model.parameters() if p.requires_grad],
                    #                                     lr=0.001, weight_decay=0.0001)
                    # scheduler_emotion_warmup = optim.lr_scheduler.ExponentialLR(optimizer_emotion_warmup, gamma=0.99)
                    # model, warmup_f1 = run_phase1_5_emotion_warmup(model, train_loader, val_loader,
                    #                                                 optimizer_emotion_warmup, scheduler_emotion_warmup,
                    #                                                 num_epochs=20, f1_threshold=0.50, patience=2, test=test)
                    # print(f"Phase 1.5 final emotion F1: {warmup_f1:.4f}")
                
                if use_personality:
                    personality_params = list(model.rawpers_fc.parameters())
                else:
                    personality_params = []  # no personality loss
                emotion_params = list(model.emotion_heads.parameters())
                # Get common shared parameters.
                # (Assuming you have a helper function get_base_params as defined in your code.)
                
                
                base_params = get_base_params(model)
                optimizer_phase2 = optim.Adam([
                    {"params": base_params, "lr": lr_base, "weight_decay": args.weight_decay_phase2},
                    {"params": personality_params, "lr": lr_person, "weight_decay": args.weight_decay_phase2},
                    {"params": emotion_params, "lr": lr_emotion, "weight_decay": args.weight_decay_phase2}
                ])
                # scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=50)
                scheduler_phase2=optim.lr_scheduler.ExponentialLR(optimizer_phase2, gamma=0.95)
                
                model, phase2_r2, phase2_f1 = run_phase2(model, train_loader, val_loader, optimizer_phase2, scheduler_phase2,
                                                    num_epochs=100, f1_threshold=0.65, patience=10, alpha=0.4,
                                                    use_personality=use_personality, test=test)
                print(f" - Phase 2 final: Personality R²: {phase2_r2:.4f}, Emotion F1: {phase2_f1:.4f}")
                model.load_state_dict(torch.load("best_model_path_phase2.pth",weights_only=True))
                if use_personality:
                    print("\n--- Phase 3: Fine-tuning Personality ---")
                    optimizer_phase3 = optim.Adam([p for p in model.parameters() if p.requires_grad],
                                                lr=2e-4, weight_decay=1e-4)
                    scheduler_phase3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase3, T_max=20)
                    model, phase3_r2 = run_phase3(model, train_loader, val_loader, optimizer_phase3, scheduler_phase3,
                                                num_epochs=30, r2_threshold=0.95, patience=5,
                                                use_personality=use_personality, test=test)
                    print(f"Phase 3 final personality R²: {phase3_r2:.4f}")
                optimizer_phase2 = optim.Adam([
                {"params": base_params, "lr": lr_base/10, "weight_decay": args.weight_decay_phase2/10},
                {"params": personality_params, "lr": lr_person/10, "weight_decay": args.weight_decay_phase2/10},
                {"params": emotion_params, "lr": lr_emotion/10, "weight_decay": args.weight_decay_phase2/10}
                ])
                # scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=50)
                # scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=200)
                scheduler_phase2=optim.lr_scheduler.ExponentialLR(optimizer_phase2, gamma=0.95)
                model, phase2_r2, phase2_f1 = run_phase2(model, train_loader, val_loader, optimizer_phase2, scheduler_phase2,
                                                    num_epochs=100, f1_threshold=0.65, patience=5, alpha=0.1,
                                                    use_personality=use_personality, test=test)
                print(f"Phase 2 final: Personality R²: {phase2_r2:.4f}, Emotion F1: {phase2_f1:.4f}")
                model.load_state_dict(torch.load("best_model_path_phase2.pth",weights_only=True))
                fold_metrics.append(phase2_f1)
                
                fold_metrics = np.array(fold_metrics)
                mean_metric = fold_metrics.mean()
                std_metric = fold_metrics.std()
                print(f"\n==== Grid {grid_id} CV Results: Mean Emotion F1 = {mean_metric:.4f} ± {std_metric:.4f} ====\n")
                
                
                ##########################################
                # Evaluate on Test Set
                ##########################################
                # Use the test_loader defined above.
                model.eval()
                all_test_person_preds, all_test_person_targets = [], []
                all_test_emotion_preds = [[] for _ in range(len(selected_emotions))]
                all_test_emotion_targets = [[] for _ in range(len(selected_emotions))]
                all_test_gender_preds = []
                all_test_gender_targets = []
                with torch.no_grad():
                    for batch in val_loader:
                        # Apply modality mask.
                        batch = apply_modality_mask(batch, modality_config)
                        eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                            x.to(device) for x in batch
                        ]
                        outputs = model(eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                                        eye_features, au_features, shimmer_features)
                        # Adjust based on your model's output signature.
                        
                        trial_delta, pred_personality, emotion_logits, _, gender_pred = outputs

                        if use_personality:
                            all_test_person_preds.append(pred_personality.cpu().numpy())
                            all_test_person_targets.append(personality_target.cpu().numpy())
                        for i in range(len(selected_emotions)):
                            preds_i = torch.argmax(emotion_logits[:, i, :], dim=1)
                            all_test_emotion_preds[i].append(preds_i.cpu().numpy())
                            all_test_emotion_targets[i].append(batch[6][:, i].cpu().numpy())
                        if use_gender and (gender_pred is not None):
                            gender_out = torch.argmax(gender_pred, dim=1)
                            all_test_gender_preds.append(gender_out.cpu().numpy())
                            all_test_gender_targets.append(gender.cpu().numpy())
                
                if use_personality:
                    test_person_preds = np.concatenate(all_test_person_preds, axis=0)
                    test_person_targets = np.concatenate(all_test_person_targets, axis=0)
                    test_r2_scores = r2_score(test_person_targets, test_person_preds, multioutput="raw_values")
                    test_r2_avg = np.mean(test_r2_scores)
                    test_r2_std = np.std(test_r2_scores)
                else:
                    test_r2_avg = None
                    test_r2_std = None
                    test_r2_scores = None

                emotion_macro_f1_list = []
                emotion_f1_list = []
                for i in range(len(selected_emotions)):
                    preds = np.concatenate(all_test_emotion_preds[i])
                    targets = np.concatenate(all_test_emotion_targets[i])
                    f1_scores=f1_score(targets, preds, average=None,zero_division=0)
                    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
                    # acc = accuracy_score(targets, preds)
                    emotion_macro_f1_list.append(macro_f1)
                    emotion_f1_list.append(f1_scores)
                emotion_f1_avg = np.mean(emotion_macro_f1_list)
                # emotion_f1_std = np.std(emotion_macro_f1_list)
                # emotion_accuracy_avg = np.mean(emotion_accuracy_list)
                # emotion_accuracy_std = np.std(emotion_accuracy_list)

                if use_gender:
                    gender_macro_f1_avg = f1_score(np.concatenate(all_test_gender_targets),
                                                   np.concatenate(all_test_gender_preds),
                                                   average="macro", zero_division=0)
                    gender_macro_f1_std = 0.0  # Single scalar value.
                else:
                    gender_macro_f1_avg = None
                    gender_macro_f1_std = None
                print(f"\n==== emotion_macro_f1_list {emotion_macro_f1_list}, test_r2_scores ==== {test_r2_scores}")
                print("===================================")
                combo_name = f"grid{grid_id}_modal_{'_'.join([k for k, v in modality_config.items() if v])}_branch_{branch_option}_stim_{use_stim_emo}"
                cv_results.append({
                    "grid_id": grid_id,
                    "modality_config": modality_config,
                    "branch_option": branch_option,
                    "use_stim_emo": use_stim_emo,
                    "personality_R2_avg": test_r2_avg,
                    "personality_R2": test_r2_scores,
                    "emotion_F1_avg": emotion_f1_avg,
                    "emotion_F1_macroes": emotion_macro_f1_list,
                    "emotion_F1":emotion_f1_list,
                    "gender_F1_avg": gender_macro_f1_avg,
                })
            
                model.eval()
                all_test_person_preds, all_test_person_targets = [], []
                all_test_emotion_preds = [[] for _ in range(len(selected_emotions))]
                all_test_emotion_targets = [[] for _ in range(len(selected_emotions))]
                all_test_gender_preds = []
                all_test_gender_targets = []
                with torch.no_grad():
                    for batch in test_loader:
                        # Apply modality mask.
                        batch = apply_modality_mask(batch, modality_config)
                        eye, pupil, au, gsr, stim_emo, personality_target, emotion_target, user_id, eye_features, au_features, shimmer_features, gender = [
                            x.to(device) for x in batch
                        ]
                        outputs = model(eye, pupil, au, gsr, stim_emo, user_id, personality_target,
                                        eye_features, au_features, shimmer_features)
                        # Adjust based on your model's output signature.
                        
                        trial_delta, pred_personality, emotion_logits, _, gender_pred = outputs

                        if use_personality:
                            all_test_person_preds.append(pred_personality.cpu().numpy())
                            all_test_person_targets.append(personality_target.cpu().numpy())
                        for i in range(len(selected_emotions)):
                            preds_i = torch.argmax(emotion_logits[:, i, :], dim=1)
                            all_test_emotion_preds[i].append(preds_i.cpu().numpy())
                            all_test_emotion_targets[i].append(batch[6][:, i].cpu().numpy())
                        if use_gender and (gender_pred is not None):
                            gender_out = torch.argmax(gender_pred, dim=1)
                            all_test_gender_preds.append(gender_out.cpu().numpy())
                            all_test_gender_targets.append(gender.cpu().numpy())
                
                if use_personality:
                    test_person_preds = np.concatenate(all_test_person_preds, axis=0)
                    test_person_targets = np.concatenate(all_test_person_targets, axis=0)
                    test_r2_scores = r2_score(test_person_targets, test_person_preds, multioutput="raw_values")
                    test_r2_avg = np.mean(test_r2_scores)
                    test_r2_std = np.std(test_r2_scores)
                else:
                    test_r2_avg = None
                    test_r2_std = None
                    test_r2_scores = None

                emotion_macro_f1_list = []
                emotion_f1_list = []
                for i in range(len(selected_emotions)):
                    preds = np.concatenate(all_test_emotion_preds[i])
                    targets = np.concatenate(all_test_emotion_targets[i])
                    f1_scores=f1_score(targets, preds, average=None,zero_division=0)
                    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
                    # acc = accuracy_score(targets, preds)
                    emotion_macro_f1_list.append(macro_f1)
                    emotion_f1_list.append(f1_scores)
                emotion_f1_avg = np.mean(emotion_macro_f1_list)
                # emotion_f1_std = np.std(emotion_macro_f1_list)
                # emotion_accuracy_avg = np.mean(emotion_accuracy_list)
                # emotion_accuracy_std = np.std(emotion_accuracy_list)

                if use_gender:
                    gender_macro_f1_avg = f1_score(np.concatenate(all_test_gender_targets),
                                                   np.concatenate(all_test_gender_preds),
                                                   average="macro", zero_division=0)
                    gender_macro_f1_std = 0.0  # Single scalar value.
                else:
                    gender_macro_f1_avg = None
                    gender_macro_f1_std = None
                print(f"\n==== emotion_macro_f1_list {emotion_macro_f1_list}, test_r2_scores ==== {test_r2_scores}")
                print("===================================")
                combo_name = f"grid{grid_id}_modal_{'_'.join([k for k, v in modality_config.items() if v])}_branch_{branch_option}_stim_{use_stim_emo}"
                save_model(model, combo_name, version="final_simple")
                results_test.append({
                    "grid_id": grid_id,
                    "modality_config": modality_config,
                    "branch_option": branch_option,
                    "use_stim_emo": use_stim_emo,
                    "personality_R2_avg": test_r2_avg,
                    "personality_R2": test_r2_scores,
                    "emotion_F1_avg": emotion_f1_avg,
                    "emotion_F1_macroes": emotion_macro_f1_list,
                    "emotion_F1":emotion_f1_list,
                    "gender_F1_avg": gender_macro_f1_avg,
                })
                grid_id += 1

    # Save overall results.
    cv_results_df = pd.DataFrame(cv_results)
    results_test_df = pd.DataFrame(results_test)
    os.makedirs("grid_results", exist_ok=True)
    cv_results_df.to_csv("grid_results/cv_results_simple.csv", index=False)
    results_test_df.to_csv("grid_results/results_test_simple.csv", index=False)
    print("\n=== 5-Fold Cross Validation and Test Set Evaluation Completed. Summary ===")
    print(cv_results_df)
    print(results_test_df)