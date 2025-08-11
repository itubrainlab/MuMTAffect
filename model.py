import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
def downsample_sequence(x, target_length=16):
    # Transpose so that the channel dimension is in position 1
    x = x.transpose(1, 2)
    # Perform adaptive average pooling directly on the current device
    x = F.adaptive_avg_pool1d(x, target_length)
    return x.transpose(1, 2)
def downsample_sequence_deterministic(x, target_length=16):
    """
    Downsamples the input sequence x deterministically by dividing the 
    sequence length into target_length equal segments and averaging over each segment.
    
    Args:
      x: Tensor of shape [B, L, D] (e.g. LSTM output).
      target_length: The desired output sequence length.
      
    Returns:
      Tensor of shape [B, target_length, D].
    """
    B, L, D = x.shape
    # Trim x so that its length is divisible by target_length.
    new_L = (L // target_length) * target_length
    if new_L == 0:
        raise ValueError("Input sequence length is smaller than target_length")
    x = x[:, :new_L, :]
    segment_length = new_L // target_length
    # Reshape so that each segment is along one new dimension.
    x = x.reshape(B, target_length, segment_length, D)
    # Take the mean over the segment dimension.
    x = x.mean(dim=2)
    return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class EpsilonInsensitiveLoss(nn.Module):
    """
    Epsilon-insensitive loss often used in support vector regression.
    
    For each prediction, if the absolute error is below epsilon, no penalty is applied.
    Otherwise, the loss is the squared error beyond epsilon.
    
    This loss is defined as:
    
      L = ((|prediction - target| - epsilon)_+)^2
      
    where (x)_+ = max(0, x).
    
    Args:
        epsilon (float): The threshold below which errors are ignored.
                         Default is 0.04.
        reduction (str): Specifies the reduction to apply to the output:
                         'mean' | 'sum' | 'none'. Default: 'mean'
    """
    def __init__(self, epsilon=0.04, reduction='mean'):
        super().__init__()
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, predictions, targets):
        # Compute the raw error between predictions and targets.
        error = predictions - targets
        
        # Optionally, check for non-finite values.
        if not torch.isfinite(error).all():
            print("Warning: 'error' tensor has non-finite values.")
        
        # Compute the absolute error.
        abs_error = torch.abs(error)
        
        # Subtract epsilon and clamp negative values to zero.
        # This makes errors within [-epsilon, epsilon] count as zero.
        adjusted_error = torch.clamp(abs_error - self.epsilon, min=0.0)
        
        # Debug: Check if adjusted_error contains any non-finite values.
        if not torch.isfinite(adjusted_error).all():
            print("Warning: 'adjusted_error' tensor has non-finite values.")
        
        loss = adjusted_error ** 2
        
        # Apply reduction: 'mean', 'sum', or return the full loss tensor.
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_emotion_loss(emotion_logits, emotion_target, emotion_class_weights, gamma=2.0, eps=1e-8):
    """
    Computes a focal loss for multiple emotion outputs.

    Parameters:
        emotion_logits: Tensor of shape [batch_size, num_emotions, num_classes]
        emotion_target: Tensor of shape [batch_size, num_emotions]
        emotion_class_weights: list of Tensors (one per emotion), each of shape [num_classes]
        gamma: Focusing parameter for focal loss.
        eps: Small epsilon value to avoid numerical issues.

    Returns:
        Averaged focal loss across emotion channels.
    """
    loss_list = []
    num_emotions = emotion_logits.shape[1]
    for i in range(num_emotions):
        # Extract logits, target, and ensure weight is on the same device.
        logits_i = torch.nan_to_num(emotion_logits[:, i, :], nan=0.0, posinf=1e6, neginf=-1e6)
        target_i = emotion_target[:, i]
        weight_i = emotion_class_weights[i].to(logits_i.device)
        
        # Compute the per-sample cross entropy loss.
        ce_loss = torch.nn.functional.cross_entropy(logits_i, target_i, weight=weight_i, reduction='none')
        # Clamp the loss to avoid extremely small values.
        ce_loss = ce_loss.clamp(min=eps)
        
        # Compute p_t safely.
        p_t = torch.exp(-ce_loss).clamp(min=eps, max=1.0 - eps)
        focal_scaling = (1 - p_t) ** gamma
        
        # For debugging: you can print tensor stats if needed.
        # print(f"Emotion {i}: ce_loss stats - mean: {ce_loss.mean().item()}, min: {ce_loss.min().item()}, max: {ce_loss.max().item()}")
        # loss_list.append((focal_scaling * ce_loss).mean())
        loss_list.append((1 * ce_loss).mean())
    
    return sum(loss_list) / len(loss_list)



# class AdvancedEmotionHead(nn.Module):
#     def __init__(self, fused_dim, persona_dim, hidden_dim, num_classes, dropout_rate,per_flag=False):
#         super(AdvancedEmotionHead, self).__init__()
#         self.per_falg=per_flag
#         if per_flag:
#             self.fc1 = nn.Linear(fused_dim + persona_dim, hidden_dim)
#         else:
#             self.fc1 = nn.Linear(fused_dim , hidden_dim)

#         self.fc2 = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, fused_repr, persona_repr):
#         if self.per_falg:
#             x = torch.cat([fused_repr, persona_repr], dim=1)
#         else:
#             x =  fused_repr
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         logits = self.fc2(x)
#         return logits

class AdvancedEmotionHead(nn.Module):
    def __init__(self, fused_dim, persona_dim, hidden_dim, num_classes, dropout_rate,per_flag=False):
        super(AdvancedEmotionHead, self).__init__()
        self.per_falg=per_flag
        
        if per_flag:
            self.fc1 = nn.Linear(fused_dim + persona_dim, hidden_dim)
            self.norm_before_fc = nn.LayerNorm(fused_dim + persona_dim)
        else:
            self.fc1 = nn.Linear(fused_dim , hidden_dim)
            self.norm_before_fc = nn.LayerNorm(fused_dim)

        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, fused_repr, persona_repr):
        if self.per_falg:
            x = torch.cat([fused_repr, persona_repr], dim=1)
        else:
            x =  fused_repr
        # x = self.norm_before_fc(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Before fc2 in your model's forward:
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("x has NaN or Inf values!")
        # print("x stats: mean {:.4f}, std {:.4f}, min {:.4f}, max {:.4f}".format(
        #     x.mean().item(), x.std().item(), x.min().item(), x.max().item()))
        # x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        # Optionally, clamp to a range:
        # x = x.clamp(min=-1e6, max=1e6)
        logits = self.fc2(x)

        return logits

class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(TemporalAttentionPooling, self).__init__()
        self.attention_fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = self.attention_fc(x)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled

### LSTM Variant
class AdvancedConcatFusionMultiModalModel(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
                 embedding_noise_std=0.0, max_val=1.0,emotion_num_labels=4):
        super(AdvancedConcatFusionMultiModalModel, self).__init__()
        hidden_dim = int(hidden_dim)
        proj_dim = int(proj_dim)
        target_length = int(target_length)
        personality_dim = int(personality_dim)
        emotion_num_classes = int(emotion_num_classes)
        transformer_nhead = int(transformer_nhead)
        transformer_layers = int(transformer_layers)
        
        self.embedding_noise_std = embedding_noise_std
        self.personality_dim = personality_dim
        self.max_val = max_val
        self.target_length = target_length
        self.finetune_mode = False

        # LSTM per modality
        self.eye_lstm    = nn.LSTM(eye_dim, hidden_dim, batch_first=True)
        self.pupil_lstm  = nn.LSTM(pupil_dim, hidden_dim, batch_first=True)
        self.au_lstm     = nn.LSTM(au_dim, hidden_dim, batch_first=True)
        self.gsr_lstm    = nn.LSTM(gsr_dim, hidden_dim, batch_first=True)

        # Project each modality's output to a lower dimension.
        self.project_modality = nn.Linear(hidden_dim, proj_dim)

        # Transformer
        self.d_model = proj_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                    nhead=transformer_nhead,
                                                    dropout=0.3,
                                                    batch_first=True)
        self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

        # Hierarchical convolutions for personality and emotion
        self.hierarchical_conv_personality = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/2]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/4]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/8]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        # self.fuse_fc_emo = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_emotion)
        # )
        self.T_emo = target_length // 8
        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(self.d_model * self.T_emo, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )
        self.d_personlity=self.d_model//2
        self.fuse_fc_per = nn.Sequential(
            nn.Linear(self.d_model,self.d_personlity ),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # Trial-level feature processing
        self.features_dim = 32
        self.eye_feat_mlp = nn.Sequential(
            nn.Linear(eye_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.au_feat_mlp = nn.Sequential(
            nn.Linear(au_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.shimmer_feat_mlp = nn.Sequential(
            nn.Linear(shimmer_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.features_mlps = nn.Sequential(
            nn.Linear(self.features_dim * 3, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )

        # Personality prediction branches
        self.emo_per_fc = nn.Sequential(
            nn.Linear(self.d_model + self.d_personlity, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion),
            nn.Linear(self.d_model, self.d_model)
        )
        self.trial_fc = nn.Sequential(
            nn.Linear(self.d_personlity + stim_emo_dim + self.features_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.rawpers_fc = nn.Sequential(
            nn.Linear(self.d_personlity+ stim_emo_dim + self.features_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_personlity + personality_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # Emotion heads (one per emotion task)
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead(self.d_model + self.features_dim+stim_emo_dim, personality_dim, self.d_model,
                                emotion_num_classes, dropout_rate=dropout_emotion)
            for _ in range()
        ])

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in,
                eye_feat, au_feat, shimmer_feat):
        # Process each modality through its LSTM, downsample and project.
        eye_out, _ = self.eye_lstm(eye)
        # eye_tok = self.project_modality(downsample_sequence(eye_out, self.target_length))
        eye_tok = self.project_modality(downsample_sequence_deterministic(eye_out, self.target_length))
        pupil_out, _ = self.pupil_lstm(pupil)
        # pupil_tok = self.project_modality(downsample_sequence(pupil_out, self.target_length))
        pupil_tok = self.project_modality(downsample_sequence_deterministic(pupil_out, self.target_length))
        au_out, _ = self.au_lstm(au)
        # au_tok = self.project_modality(downsample_sequence(au_out, self.target_length))
        au_tok = self.project_modality(downsample_sequence_deterministic(au_out, self.target_length))
        gsr_out, _ = self.gsr_lstm(gsr)
        # gsr_tok = self.project_modality(downsample_sequence(gsr_out, self.target_length))
        gsr_tok = self.project_modality(downsample_sequence_deterministic(gsr_out, self.target_length))

        mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)
        mod_tokens = mod_tokens + self.pos_embedding
        mod_tokens = self.modal_transformer(mod_tokens)
        x = mod_tokens.transpose(1, 2)  # Shape: [B, d_model, T]

        # Personality branch
        x_per = self.hierarchical_conv_personality(x)
        fused_per = self.fuse_fc_per(x_per.mean(dim=2))

        # Emotion branch (updated: flatten instead of mean)
        x_emo = self.hierarchical_conv_emotion(x)
        # Instead of taking the mean over the temporal dimension, flatten all features:
        x_emo_flat = x_emo.flatten(start_dim=1)  # Shape: [B, d_model * T_emo]
        fused_emo = self.fuse_fc_emo(x_emo_flat)   # Shape: [B, d_model
        fused_emo=self.emo_per_fc(torch.cat([fused_per, fused_emo], dim=1))
        # Process trial-level features
        eye_trial = self.eye_feat_mlp(eye_feat)
        au_trial = self.au_feat_mlp(au_feat)
        shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
        trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
        trial_features = self.features_mlps(trial_concat)

        # Personality prediction
        if stim_emo is not None:
            concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
        else:
            concat_in = torch.cat([fused_per, trial_features], dim=1)
        pred_personality = self.rawpers_fc(concat_in)
        if self.finetune_mode:
            trial_delta = self.trial_fc(concat_in)
            trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
        else:
            trial_delta = torch.zeros_like(pred_personality)
        golden_personality = pred_personality + trial_delta

        # Fusion for emotion prediction
        # advanced_persona = self.fusion_mlp(fusion_input)
        concat_in_emo = torch.cat([fused_emo, trial_features,stim_emo], dim=1)
        emotion_logits_list = []
        for head in self.emotion_heads:
            logits = head(concat_in_emo, golden_personality)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)
        return trial_delta, pred_personality, emotion_logits, fused_per


class AdvancedConcatFusionMultiModalModelGRU(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.3, dropout_emotion=0.25, dropout_features=0.3,
                 embedding_noise_std=0.0, max_val=1.0,emotion_num_labels=2):
        super(AdvancedConcatFusionMultiModalModelGRU, self).__init__()
        hidden_dim = int(hidden_dim)
        proj_dim = int(proj_dim)
        target_length = int(target_length)
        personality_dim = int(personality_dim)
        emotion_num_classes = int(emotion_num_classes)
        transformer_nhead = int(transformer_nhead)
        transformer_layers = int(transformer_layers)
        self.emotion_num_labels = emotion_num_labels
        self.embedding_noise_std = embedding_noise_std
        self.personality_dim = personality_dim
        self.max_val = max_val
        self.target_length = target_length
        self.finetune_mode = False

        # GRU per modality (replacing LSTMs)
        self.eye_gru    = nn.GRU(eye_dim, hidden_dim, batch_first=True)
        self.pupil_gru  = nn.GRU(pupil_dim, hidden_dim, batch_first=True)
        self.au_gru     = nn.GRU(au_dim, hidden_dim, batch_first=True)
        self.gsr_gru    = nn.GRU(gsr_dim, hidden_dim, batch_first=True)

        # Project each modality's output to a lower dimension.
        self.project_modality = nn.Linear(hidden_dim, proj_dim)

        # Transformer
        self.d_model = proj_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                    nhead=transformer_nhead,
                                                    dropout=0.2,
                                                    batch_first=True)
        self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

        # Hierarchical convolutions for personality and emotion
        self.hierarchical_conv_personality = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/2]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/4]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/8]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        self.T_emo = target_length // 8
        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(self.d_model * self.T_emo, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )
        self.d_personlity = self.d_model // 2
        self.fuse_fc_per = nn.Sequential(
            nn.Linear(self.d_model, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # Trial-level feature processing
        self.features_dim = 32
        self.eye_feat_mlp = nn.Sequential(
            nn.Linear(eye_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.au_feat_mlp = nn.Sequential(
            nn.Linear(au_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.shimmer_feat_mlp = nn.Sequential(
            nn.Linear(shimmer_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.features_mlps = nn.Sequential(
            nn.Linear(self.features_dim * 3, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )

        # Personality prediction branches
        self.emo_per_fc = nn.Sequential(
            nn.Linear(self.d_model + self.d_personlity, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion),
            nn.Linear(self.d_model, self.d_model)
        )
        self.trial_fc = nn.Sequential(
            nn.Linear(self.d_personlity + stim_emo_dim + self.features_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.rawpers_fc = nn.Sequential(
            nn.Linear(self.d_personlity + stim_emo_dim + self.features_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_personlity + personality_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # Emotion heads (one per emotion task)
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead(self.d_model + self.features_dim, personality_dim, self.d_model,
                                emotion_num_classes, dropout_rate=dropout_emotion)
            for _ in range(emotion_num_labels)
        ])

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in,
                eye_feat, au_feat, shimmer_feat):
        # Process each modality through its GRU, downsample and project.
        eye_out, _ = self.eye_gru(eye)
        eye_tok = self.project_modality(downsample_sequence_deterministic(eye_out, self.target_length))
        pupil_out, _ = self.pupil_gru(pupil)
        pupil_tok = self.project_modality(downsample_sequence_deterministic(pupil_out, self.target_length))
        au_out, _ = self.au_gru(au)
        au_tok = self.project_modality(downsample_sequence_deterministic(au_out, self.target_length))
        gsr_out, _ = self.gsr_gru(gsr)
        gsr_tok = self.project_modality(downsample_sequence_deterministic(gsr_out, self.target_length))

        mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)
        mod_tokens = mod_tokens + self.pos_embedding
        mod_tokens = self.modal_transformer(mod_tokens)
        x = mod_tokens.transpose(1, 2)

        # Personality branch
        x_per = self.hierarchical_conv_personality(x)
        fused_per = self.fuse_fc_per(x_per.mean(dim=2))

        # Emotion branch (updated: flatten instead of mean)
        x_emo = self.hierarchical_conv_emotion(x)
        # Instead of taking the mean over the temporal dimension, flatten all features:
        x_emo_flat = x_emo.flatten(start_dim=1)  # Shape: [B, d_model * T_emo]
        fused_emo = self.fuse_fc_emo(x_emo_flat)   # Shape: [B, d_model]
        fused_emo = self.emo_per_fc(torch.cat([fused_per, fused_emo], dim=1))
        
        # Process trial-level features
        eye_trial = self.eye_feat_mlp(eye_feat)
        au_trial = self.au_feat_mlp(au_feat)
        shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
        trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
        trial_features = self.features_mlps(trial_concat)

        # Personality prediction
        if stim_emo is not None:
            concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
        else:
            concat_in = torch.cat([fused_per, trial_features], dim=1)
        pred_personality = self.rawpers_fc(concat_in)
        if self.finetune_mode:
            trial_delta = self.trial_fc(concat_in)
            trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
        else:
            trial_delta = torch.zeros_like(pred_personality)
        golden_personality = pred_personality + trial_delta

        # Fusion for emotion prediction
        fusion_input = torch.cat([fused_per, golden_personality], dim=1)
        concat_in_emo = torch.cat([fused_emo, trial_features], dim=1)
        emotion_logits_list = []
        for head in self.emotion_heads:
            logits = head(concat_in_emo, golden_personality)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)
        return trial_delta, pred_personality, emotion_logits, fused_per
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, L, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


def scale_with_sigmoid(x, max_val):
    return (torch.sigmoid(x) - 0.5) * 0.1 * max_val

class AdvancedConcatFusionMultiModalModelTransformer(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
                 embedding_noise_std=0.0, max_val=1.0):
        super(AdvancedConcatFusionMultiModalModelTransformer, self).__init__()
        # Note: Although hidden_dim is passed, we use a smaller dimension for the modality transformer.
        hidden_dim = int(hidden_dim)
        proj_dim = int(proj_dim)
        target_length = int(target_length)
        personality_dim = int(personality_dim)
        emotion_num_classes = int(emotion_num_classes)
        transformer_nhead = int(transformer_nhead)
        transformer_layers = int(transformer_layers)
        
        self.embedding_noise_std = embedding_noise_std
        self.personality_dim = personality_dim
        self.max_val = max_val
        self.target_length = target_length
        self.finetune_mode = False
        # Define the dimension for the small transformer
        self.transformer_small_dim = 64  
        self.transformer_small_head = 2

        # Replace modality LSTMs/GRUs with small transformers.
        # Project raw inputs to transformer_small_dim instead of hidden_dim.
        self.eye_proj    = nn.Linear(eye_dim, self.transformer_small_dim)
        self.pupil_proj  = nn.Linear(pupil_dim, self.transformer_small_dim)
        self.au_proj     = nn.Linear(au_dim, self.transformer_small_dim)
        self.gsr_proj    = nn.Linear(gsr_dim, self.transformer_small_dim)
        
        # Shared positional encoding for modalities.
        self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)
        
        # Transformer for each modality (using 1 layer for a "small" transformer)
        self.eye_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.2, batch_first=True),
            num_layers=1
        )
        self.pupil_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.2, batch_first=True),
            num_layers=1
        )
        self.au_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.2, batch_first=True),
            num_layers=1
        )
        self.gsr_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.2, batch_first=True),
            num_layers=1
        )

        # Project each modality's output to a lower dimension.
        self.project_modality = nn.Linear(self.transformer_small_dim, proj_dim)

        # Transformer for modality fusion remains unchanged.
        self.d_model = proj_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                    nhead=transformer_nhead,
                                                    dropout=0.2,
                                                    batch_first=True)
        self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

        # Hierarchical convolutions for personality and emotion
        self.hierarchical_conv_personality = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/2]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/4]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),  # Output: [B, d_model, T/8]
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        # L_in = target_length + personality_dim  # this is the length after concatenating personality tokens
        # L1 = math.floor((L_in + 2 - 3) / 2) + 1
        # L2 = math.floor((L1 + 2 - 3) / 2) + 1
        # L3 = math.floor((L2 + 2 - 3) / 2) + 1
        # self.T_emo = L3
        # self.fuse_fc_emo = nn.Sequential(
        #     nn.Linear(self.d_model * self.T_emo, self.d_model),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_emotion)
        # )
        self.fixed_T_emo = 10  # fixed number of temporal tokens after pooling for the emotion branch
        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(self.d_model * self.fixed_T_emo, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )


        self.d_personlity = self.d_model // 2
        self.fuse_fc_per = nn.Sequential(
            nn.Linear(self.d_model, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # Trial-level feature processing
        self.features_dim = 32
        self.eye_feat_mlp = nn.Sequential(
            nn.Linear(eye_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.au_feat_mlp = nn.Sequential(
            nn.Linear(au_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.shimmer_feat_mlp = nn.Sequential(
            nn.Linear(shimmer_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.features_mlps = nn.Sequential(
            nn.Linear(self.features_dim * 3, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.fuse_personality = nn.Linear(2 * self.d_model, self.d_model)
        self.personality_token_proj = nn.Linear(1, self.d_model)

        # Personality prediction branches
        self.emo_per_fc = nn.Sequential(
            nn.Linear(self.d_model + self.d_personlity, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion),
            nn.Linear(self.d_model, self.d_model)
        )
        self.trial_fc = nn.Sequential(
            nn.Linear(self.d_personlity + stim_emo_dim + self.features_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.rawpers_fc = nn.Sequential(
            nn.Linear(self.d_personlity + stim_emo_dim + self.features_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_personlity + personality_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # Emotion heads (one per emotion task)
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead(self.d_model + self.features_dim, personality_dim, self.d_model,
                                emotion_num_classes, dropout_rate=dropout_emotion)
            for _ in range(4)
        ])

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in,
            eye_feat, au_feat, shimmer_feat):
        # --- Process modalities ---
        # Eye modality
        eye_emb = self.eye_proj(eye)
        eye_emb = self.pos_encoding_modality(eye_emb)
        eye_out = self.eye_transformer(eye_emb)
        eye_tok = self.project_modality(downsample_sequence_deterministic(eye_out, self.target_length))
        
        # Pupil modality
        pupil_emb = self.pupil_proj(pupil)
        pupil_emb = self.pos_encoding_modality(pupil_emb)
        pupil_out = self.pupil_transformer(pupil_emb)
        pupil_tok = self.project_modality(downsample_sequence_deterministic(pupil_out, self.target_length))
        
        # AU modality
        au_emb = self.au_proj(au)
        au_emb = self.pos_encoding_modality(au_emb)
        au_out = self.au_transformer(au_emb)
        au_tok = self.project_modality(downsample_sequence_deterministic(au_out, self.target_length))
        
        # GSR modality
        gsr_emb = self.gsr_proj(gsr)
        gsr_emb = self.pos_encoding_modality(gsr_emb)
        gsr_out = self.gsr_transformer(gsr_emb)
        gsr_tok = self.project_modality(downsample_sequence_deterministic(gsr_out, self.target_length))
        
        # Concatenate modality tokens and add positional embedding.
        raw_mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)  # [B, T, d_model]
        raw_mod_tokens = raw_mod_tokens + self.pos_embedding
        
        # --- Personality branch ---
        # Process the fused modality tokens via hierarchical convolutions.
        x = raw_mod_tokens.transpose(1, 2)  # [B, d_model, T]
        x_per = self.hierarchical_conv_personality(x)  # [B, d_model, T_per] (T_per < T)
        fused_per = self.fuse_fc_per(x_per.mean(dim=2))  # [B, d_personlity]
        
        # Upsample x_per to match raw_mod_tokens temporal length T.
        x_per_upsampled = torch.nn.functional.interpolate(
            x_per, size=raw_mod_tokens.size(1), mode='linear', align_corners=False)
        x_per_upsampled = x_per_upsampled.transpose(1, 2)  # [B, T, d_model]
        
        # Fuse raw modality tokens with upsampled personality branch features.
        fused_tokens = torch.cat([raw_mod_tokens, x_per_upsampled], dim=-1)  # [B, T, 2*d_model]
        fused_tokens = self.fuse_personality(fused_tokens)  # [B, T, d_model]
        
        # --- Trial-level features & Personality Prediction ---
        eye_trial = self.eye_feat_mlp(eye_feat)
        au_trial = self.au_feat_mlp(au_feat)
        shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
        trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
        trial_features = self.features_mlps(trial_concat)
        
        if stim_emo is not None:
            concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
        else:
            concat_in = torch.cat([fused_per, trial_features], dim=1)
        pred_personality = self.rawpers_fc(concat_in)  # [B, personality_dim]
        
        if self.finetune_mode:
            trial_delta = self.trial_fc(concat_in)
            trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
        else:
            trial_delta = torch.zeros_like(pred_personality)
        golden_personality = pred_personality + trial_delta

        # --- Project personality prediction into tokens ---
        # Map each scalar personality score to a d_model-dimensional token.
        personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
        # personality_tokens: [B, personality_dim, d_model]

        # --- Boost and Prepend Personality Tokens ---
        scale_factor = 2.0  # Adjust to boost their influence.
        personality_tokens = scale_factor * personality_tokens
        
        replication_factor = 3  # Replicate tokens to increase their presence.
        personality_tokens = personality_tokens.repeat(1, replication_factor, 1)
        # Now personality_tokens: [B, personality_dim * replication_factor, d_model]
        
        # Prepend personality tokens to the fused modality tokens.
        fused_tokens_with_personality = torch.cat([personality_tokens, fused_tokens], dim=1)
        # fused_tokens_with_personality: [B, (replicated_personality_tokens + T), d_model]
        
        # Feed the entire sequence into the modal_transformer.
        mod_tokens = self.modal_transformer(fused_tokens_with_personality)
        
        # --- Emotion branch using adaptive pooling ---
        x_emo = self.hierarchical_conv_emotion(mod_tokens.transpose(1, 2))
        # Use adaptive pooling to force a fixed output size regardless of input sequence length.
        x_emo_pooled = torch.nn.functional.adaptive_avg_pool1d(x_emo, output_size=self.fixed_T_emo)
        x_emo_flat = x_emo_pooled.flatten(start_dim=1)  # [B, d_model * fixed_T_emo]
        fused_emo = self.fuse_fc_emo(x_emo_flat)   # [B, d_model]
        fused_emo = self.emo_per_fc(torch.cat([fused_per, fused_emo], dim=1))
        
        # --- Emotion prediction ---
        concat_in_emo = torch.cat([fused_emo, trial_features], dim=1)
        emotion_logits_list = []
        for head in self.emotion_heads:
            logits = head(concat_in_emo, golden_personality)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)
        
        return trial_delta, pred_personality, emotion_logits, fused_per

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# (Keep your existing helper functions, PositionalEncoding, etc.)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------
# PositionalEncoding and TaskAttention remain unchanged.
# -----------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TaskAttention(nn.Module):
    def __init__(self, d_model):
        super(TaskAttention, self).__init__()
        # Learnable queries for each task.
        self.personality_query = nn.Parameter(torch.randn(1, d_model))
        self.emotion_query = nn.Parameter(torch.randn(1, d_model))
    
    def forward(self, tokens):
        # tokens: [B, T, d_model]
        B, T, d = tokens.shape
        q_person = self.personality_query.expand(B, 1, d)  # [B, 1, d]
        q_emotion = self.emotion_query.expand(B, 1, d)      # [B, 1, d]
        attn_person = torch.softmax(torch.bmm(q_person, tokens.transpose(1,2)) / math.sqrt(d), dim=-1)  # [B, 1, T]
        rep_person = torch.bmm(attn_person, tokens)  # [B, 1, d]
        attn_emotion = torch.softmax(torch.bmm(q_emotion, tokens.transpose(1,2)) / math.sqrt(d), dim=-1)  # [B, 1, T]
        rep_emotion = torch.bmm(attn_emotion, tokens)  # [B, 1, d]
        return rep_person.squeeze(1), rep_emotion.squeeze(1)  # each: [B, d]
class TaskAttentionTemporal(nn.Module):
    def __init__(self, d_model):
        super(TaskAttentionTemporal, self).__init__()
        # Separate linear layers for the two query streams
        self.metadata_q_proj = nn.Linear(d_model, d_model)
        self.trial_q_proj = nn.Linear(d_model, d_model)
        # Shared key and value projections (you can also separate these if needed)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens):
        # tokens: [B, T, d_model]
        B, T, d = tokens.shape
        
        # Compute keys and values from tokens.
        K = self.k_proj(tokens)  # shape: [B, T, d_model]
        V = self.v_proj(tokens)  # shape: [B, T, d_model]
        
        # For metadata stream: use one query projection.
        q_metadata = self.metadata_q_proj(tokens)  # [B, T, d_model]
        # Compute attention weights: each token attends to all tokens.
        attn_weights_metadata = torch.softmax(
            torch.bmm(q_metadata, K.transpose(1, 2)) / self.scale, dim=-1
        )  # [B, T, T]
        # Aggregate values for metadata stream.
        rep_metadata = torch.bmm(attn_weights_metadata, V)  # [B, T, d_model]
        
        # For trial-level stream: use the other query projection.
        q_trial = self.trial_q_proj(tokens)  # [B, T, d_model]
        attn_weights_trial = torch.softmax(
            torch.bmm(q_trial, K.transpose(1, 2)) / self.scale, dim=-1
        )  # [B, T, T]
        rep_trial = torch.bmm(attn_weights_trial, V)  # [B, T, d_model]
        
        return rep_metadata, rep_trial
# -----------------------
# Modified Model with optional gender prediction and optional use of stim_emo.
# -----------------------

class AdvancedConcatFusionMultiModalModelTransformerAttention(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
                 embedding_noise_std=0.0, max_val=1.0,
                 emotion_num_labels=4,
                 use_gender=False,        # NEW: flag for gender branch
                 use_stim_emo=True,use_personality=True):      # NEW: flag to use stim_emo input
        super(AdvancedConcatFusionMultiModalModelTransformerAttention, self).__init__()
        # Cast to int.
        hidden_dim = int(hidden_dim)
        proj_dim = int(proj_dim)
        target_length = int(target_length)
        personality_dim = int(personality_dim)
        emotion_num_classes = int(emotion_num_classes)
        transformer_nhead = int(transformer_nhead)
        transformer_layers = int(transformer_layers)
        self.stim_emo_dim = stim_emo_dim
        self.embedding_noise_std = embedding_noise_std
        self.personality_dim = personality_dim
        self.max_val = max_val
        self.target_length = target_length
        self.finetune_mode = False
        self.emotion_num_labels = emotion_num_labels
        self.use_gender = use_gender          # NEW flag
        self.use_stim_emo = use_stim_emo      # NEW flag
        self.use_personality = use_personality # NEW flag
        # Define dimension for small modality transformers.
        self.transformer_small_dim = 64  
        self.transformer_small_head = 2

        # Project raw inputs.
        self.eye_proj    = nn.Linear(eye_dim, self.transformer_small_dim)
        self.pupil_proj  = nn.Linear(pupil_dim, self.transformer_small_dim)
        self.au_proj     = nn.Linear(au_dim, self.transformer_small_dim)
        self.gsr_proj    = nn.Linear(gsr_dim, self.transformer_small_dim)
        
        self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)
        
        # Modality-specific transformers.
        self.eye_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.3, batch_first=True),
            num_layers=1
        )
        self.pupil_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.3, batch_first=True),
            num_layers=1
        )
        self.au_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.3, batch_first=True),
            num_layers=1
        )
        self.gsr_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.3, batch_first=True),
            num_layers=1
        )
        self.project_modality = nn.Linear(self.transformer_small_dim, proj_dim)

        # Transformer for modality fusion.
        self.d_model = proj_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                    nhead=transformer_nhead,
                                                    dropout=0.3,
                                                    batch_first=True)
        self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

        # Hierarchical convolutions.
        self.hierarchical_conv_personality = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        if stim_emo_dim == 0:
            self.fixed_T_emo = 8   # new value when stim_emo is disabled
        else:
            self.fixed_T_emo = 10  # default value when stim_emo is used
        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(self.d_model * self.fixed_T_emo, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )

        self.d_personlity = self.d_model // 2
        self.fuse_fc_per = nn.Sequential(
            nn.Linear(self.d_model, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # --- Task Attention for Personality vs. Emotion ---
        self.task_attention = TaskAttention(d_model=self.d_model)
        self.personality_attention_proj = nn.Linear(self.d_model, self.d_personlity)

        # Trial-level features.
        self.features_dim = 32
        self.eye_feat_mlp = nn.Sequential(
            nn.Linear(eye_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.au_feat_mlp = nn.Sequential(
            nn.Linear(au_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.shimmer_feat_mlp = nn.Sequential(
            nn.Linear(shimmer_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.features_mlps = nn.Sequential(
            nn.Linear(self.features_dim * 3, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.fuse_personality = nn.Linear(2 * self.d_model, self.d_model)
        self.personality_token_proj = nn.Linear(1, self.d_model)

        # Personality prediction.
        # Update emo_per_fc input to account for additional attention signal.
        self.emo_per_fc = nn.Sequential(
            nn.Linear(self.d_personlity + 2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion),
            nn.Linear(self.d_model, self.d_model)
        )
        self.trial_fc = nn.Sequential(
            nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.rawpers_fc = nn.Sequential(
            nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_personlity + personality_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # --- Optional Gender Branch ---
        if self.use_gender:
            # Input dimension: same as concat_in used for personality prediction.
            gender_input_dim = self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0)
            gender_hidden_dim = 32  # you can tune this value
            self.gender_branch = nn.Sequential(
                nn.Linear(gender_input_dim, gender_hidden_dim),
                nn.ReLU(),
                nn.Linear(gender_hidden_dim, 2)
            )
        else:
            self.gender_branch = None

        # Emotion heads.
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead(self.d_model + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), personality_dim, self.d_model,
                                emotion_num_classes, dropout_rate=dropout_emotion)
            for _ in range(self.emotion_num_labels)
        ])

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in,
                eye_feat, au_feat, shimmer_feat):
        # --- Use or ignore stim_emo based on flag ---
        if not self.use_stim_emo:
            stim_emo = None

        # --- Process modalities ---
        # Eye modality
        eye_emb = self.eye_proj(eye)
        eye_emb = self.pos_encoding_modality(eye_emb)
        eye_out = self.eye_transformer(eye_emb)
        eye_tok = self.project_modality(downsample_sequence_deterministic(eye_out, self.target_length))
        # Pupil modality
        pupil_emb = self.pupil_proj(pupil)
        pupil_emb = self.pos_encoding_modality(pupil_emb)
        pupil_out = self.pupil_transformer(pupil_emb)
        pupil_tok = self.project_modality(downsample_sequence_deterministic(pupil_out, self.target_length))
        # AU modality
        au_emb = self.au_proj(au)
        au_emb = self.pos_encoding_modality(au_emb)
        au_out = self.au_transformer(au_emb)
        au_tok = self.project_modality(downsample_sequence_deterministic(au_out, self.target_length))
        # GSR modality
        gsr_emb = self.gsr_proj(gsr)
        gsr_emb = self.pos_encoding_modality(gsr_emb)
        gsr_out = self.gsr_transformer(gsr_emb)
        gsr_tok = self.project_modality(downsample_sequence_deterministic(gsr_out, self.target_length))
        
        raw_mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)  # [B, T, d_model]
        raw_mod_tokens = raw_mod_tokens + self.pos_embedding
        
        # --- Personality branch ---
        x = raw_mod_tokens.transpose(1, 2)  # [B, d_model, T]
        x_per = self.hierarchical_conv_personality(x)  # [B, d_model, T_per]
        fused_per = self.fuse_fc_per(x_per.mean(dim=2))  # [B, d_personlity]
        
        x_per_upsampled = F.interpolate(x_per, size=raw_mod_tokens.size(1), mode='linear', align_corners=False)
        x_per_upsampled = x_per_upsampled.transpose(1, 2)  # [B, T, d_model]
        if not self.use_personality:
            x_per_upsampled = torch.zeros_like(x_per_upsampled)
        fused_tokens = torch.cat([raw_mod_tokens, x_per_upsampled], dim=-1)  # [B, T, 2*d_model]
        fused_tokens = self.fuse_personality(fused_tokens)  # [B, T, d_model]
        
        # --- Task Attention ---
        rep_person, rep_emotion = self.task_attention(fused_tokens)  # each: [B, d_model]
        rep_person_proj = self.personality_attention_proj(rep_person)  # [B, d_personlity]
        final_person_rep = fused_per + rep_person_proj  # [B, d_personlity]
        
        # --- Trial-level features & Personality Prediction ---
        eye_trial = self.eye_feat_mlp(eye_feat)
        au_trial = self.au_feat_mlp(au_feat)
        shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
        trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
        trial_features = self.features_mlps(trial_concat)
        if self.use_personality:
            if self.use_stim_emo:
                # Include stim_emo: expected total dims = 64+32+6 = 102.
                concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
            else:
                # Exclude stim_emo: expected total dims = 64+32 = 96.
                concat_in = torch.cat([fused_per, trial_features], dim=1)
            pred_personality = self.rawpers_fc(concat_in)
            if self.finetune_mode:
                trial_delta = self.trial_fc(concat_in)
                trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
            else:
                trial_delta = torch.zeros_like(pred_personality)
            golden_personality = pred_personality + trial_delta
        else:
            # If personality is not used, return zeros.
            batch_size = fused_per.size(0)
            pred_personality = torch.zeros(batch_size, self.personality_dim, device=fused_per.device)
            trial_delta = torch.zeros_like(pred_personality)
            golden_personality = pred_personality

        # --- Optional Gender Prediction ---
        gender_pred = None
        if self.use_gender:
            gender_pred = self.gender_branch(concat_in)
        
        # --- Project personality prediction into tokens ---
        personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
        scale_factor = 2.0
        personality_tokens = scale_factor * personality_tokens
        replication_factor = 3
        personality_tokens = personality_tokens.repeat(1, replication_factor, 1)
        fused_tokens_with_personality = torch.cat([personality_tokens, fused_tokens], dim=1)
        mod_tokens = self.modal_transformer(fused_tokens_with_personality)
        
        # --- Emotion branch ---
        x_emo = self.hierarchical_conv_emotion(mod_tokens.transpose(1, 2))
        # Use CPU workaround for adaptive pooling on MPS.
        x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)
        x_emo_flat = x_emo_pooled.flatten(start_dim=1)
        fused_emo = self.fuse_fc_emo(x_emo_flat)
        fused_emo = self.emo_per_fc(torch.cat([final_person_rep, fused_emo, rep_emotion], dim=1))
        
        if stim_emo is not None:
            concat_in_emo = torch.cat([fused_emo, trial_features, stim_emo], dim=1)
        else:
            concat_in_emo = torch.cat([fused_emo, trial_features], dim=1)
        emotion_logits_list = []
        for head in self.emotion_heads:
            logits = head(concat_in_emo, golden_personality)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)
        
        return trial_delta, pred_personality, emotion_logits, fused_per, gender_pred
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedEmotionHead_v2(nn.Module):
    def __init__(self, fused_dim, persona_dim, stim_emo_dim,hidden_dim, num_classes, dropout_rate, per_flag=False):
        super(AdvancedEmotionHead_v2, self).__init__()
        self.per_flag = per_flag

        # Dimension used by attention. 
        # If persona is used, the "token" dimension is fused_dim + persona_dim
        # else it is fused_dim only.
        self.attn_dim = fused_dim if not per_flag else (fused_dim + persona_dim)

        # A single-head self-attention over the tokens.
        # You could use more heads if you like, but 1 is simplest.
        self.self_attn = nn.MultiheadAttention(embed_dim=self.attn_dim, 
                                               num_heads=1, 
                                               batch_first=True)
        # Optional layer norm after attention
        self.attn_norm = nn.LayerNorm(self.attn_dim)

        # Keep the same linear layers as before
        if per_flag:
            # Input to fc1 is dimension = fused_dim + persona_dim
            self.fc1 = nn.Linear(fused_dim + persona_dim, hidden_dim)
            self.norm_before_fc = nn.LayerNorm(fused_dim + persona_dim)
        else:
            self.fc1 = nn.Linear(fused_dim, hidden_dim)
            self.norm_before_fc = nn.LayerNorm(fused_dim)

        self.fc2 = nn.Linear(hidden_dim+stim_emo_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, fused_repr, persona_repr,stim_emo=None):
        """
        fused_repr:  [batch_size, fused_dim]
        persona_repr: [batch_size, persona_dim] (may be unused if per_flag=False)
        """

        # ---------------------------------------------------------
        # 1) Prepare tokens for attention
        #    If persona is used, we have two tokens. Otherwise, just one.
        # ---------------------------------------------------------
        if self.per_flag:
            # shape: (B, 2, d_attn) where d_attn = fused_dim + persona_dim
            # We simply "stack" them along a new dimension=1 so self-attn sees them as a sequence of length 2.
            x = torch.cat([fused_repr, persona_repr], dim=1)
            # x shape: (B, fused_dim + persona_dim)
            # Turn it into "tokens" by unsqueezing for a seq dimension.
            x = x.unsqueeze(1)  # => (B, 1, fused_dim+persona_dim)

            # But we actually want them as separate tokens: 
            #   token 1 = fused_repr, token 2 = persona_repr
            # So let's stack them properly:
            #   fused_repr -> (B, 1, fused_dim)
            #   persona_repr -> (B, 1, persona_dim)
            #   Then cat => (B, 1, fused_dim) + (B, 1, persona_dim) => (B, 1, fused_dim + persona_dim)
            # This is effectively one "token" though. If we truly want 2 tokens,
            # we do:
            fused_token = fused_repr.unsqueeze(1)   # (B, 1, fused_dim)
            persona_token = persona_repr.unsqueeze(1)  # (B, 1, persona_dim)
            # Now we pad them out so each has shape (B,1, fused_dim+persona_dim) or we could keep them separate.
            # Easiest: stack directly along dim=1, but they must match dimension along dim=2. 
            # Alternatively, we can do single self-attn over a 2-token sequence if we first project each token to the same size. 

            # For simplicity, let's do a single "concat-based token," but that doesn't give us real 2-token attention.
            # Let's do "true 2-token attention" by projecting each to attn_dim, then stacking:
            fused_proj = torch.cat([fused_repr, torch.zeros_like(persona_repr)], dim=1) # shape (B, fused_dim+persona_dim)
            persona_proj = torch.cat([torch.zeros_like(fused_repr), persona_repr], dim=1) # shape (B, fused_dim+persona_dim)
            tokens = torch.stack([fused_proj, persona_proj], dim=1) # shape (B,2,fused_dim+persona_dim)

        else:
            # If no persona, we only have "fused_repr" as a single token
            # shape: (B,1,fused_dim)
            tokens = fused_repr.unsqueeze(1)

        # ---------------------------------------------------------
        # 2) Apply single-head attention: 
        #    Q=K=V = tokens => self-attention among these tokens.
        # ---------------------------------------------------------
        # tokens shape: (B, seq_len, attn_dim), with seq_len=2 if per_flag, else 1
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        # shape of attn_out = (B, seq_len, attn_dim)

        # We can pool (average or sum) across the sequence dimension 
        # to get a single vector. For seq_len=1, it’s trivial.
        fused_attn = attn_out.mean(dim=1)  # shape (B, attn_dim)

        # Optional norm after attention:
        fused_attn = self.attn_norm(fused_attn)

        # ---------------------------------------------------------
        # 3) Same final MLP as before (fc1->relu->dropout->fc2)
        # ---------------------------------------------------------
        # x = self.norm_before_fc(fused_attn)  # optional pre-LayerNorm, up to you
        x = F.relu(self.fc1(fused_attn))
        x = self.dropout(x)
        if stim_emo != None:
            x= torch.cat([x, stim_emo], dim=1) 
        # Check for NaN or Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: x has NaN or Inf values in AdvancedEmotionHead forward!")

        logits = self.fc2(x)
        return logits


# class AdvancedConcatFusionMultiModalModelTransformerAttention(nn.Module):
#     def __init__(self,
#                  eye_dim, pupil_dim, au_dim, gsr_dim,
#                  stim_emo_dim, hidden_dim, target_length,
#                  personality_dim, emotion_num_classes, num_users,
#                  eye_feat_dim, au_feat_dim, shimmer_feat_dim,
#                  proj_dim=32, transformer_nhead=4, transformer_layers=1,
#                  dropout_person=0.4, dropout_emotion=0.25, dropout_features=0.3,
#                  embedding_noise_std=0.0, max_val=1.0,
#                  emotion_num_labels=4,
#                  use_gender=False,        # NEW: flag for gender branch
#                  use_stim_emo=True,use_personality=True):      # NEW: flag to use stim_emo input
#         super(AdvancedConcatFusionMultiModalModelTransformerAttention, self).__init__()
#         # Cast to int.
#         hidden_dim = int(hidden_dim)
#         proj_dim = int(proj_dim)
#         target_length = int(target_length)
#         personality_dim = int(personality_dim)
#         emotion_num_classes = int(emotion_num_classes)
#         transformer_nhead = int(transformer_nhead)
#         transformer_layers = int(transformer_layers)
#         self.stim_emo_dim = stim_emo_dim
#         self.embedding_noise_std = embedding_noise_std
#         self.personality_dim = personality_dim
#         self.max_val = max_val
#         self.target_length = target_length
#         self.finetune_mode = False
#         self.emotion_num_labels = emotion_num_labels
#         self.use_gender = use_gender          # NEW flag
#         self.use_stim_emo = use_stim_emo      # NEW flag
#         self.use_personality = use_personality # NEW flag
#         # Define dimension for small modality transformers.
#         self.transformer_small_dim = 64  
#         self.transformer_small_head = 2

#         # Project raw inputs.
#         self.eye_proj    = nn.Linear(eye_dim, self.transformer_small_dim)
#         self.pupil_proj  = nn.Linear(pupil_dim, self.transformer_small_dim)
#         self.au_proj     = nn.Linear(au_dim, self.transformer_small_dim)
#         self.gsr_proj    = nn.Linear(gsr_dim, self.transformer_small_dim)
        
#         self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)
#         transformer_dropout=0.25
#         # Modality-specific transformers.
#         self.eye_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
#                                        dropout=transformer_dropout, batch_first=True),
#             num_layers=1
#         )
#         self.pupil_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
#                                        dropout=transformer_dropout, batch_first=True),
#             num_layers=1
#         )
#         self.au_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
#                                        dropout=transformer_dropout, batch_first=True),
#             num_layers=1
#         )
#         self.gsr_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
#                                        dropout=transformer_dropout, batch_first=True),
#             num_layers=1
#         )
#         self.project_modality = nn.Linear(self.transformer_small_dim, proj_dim)

#         # Transformer for modality fusion.
#         self.d_model = proj_dim * 4
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
#                                                     nhead=transformer_nhead,
#                                                     dropout=transformer_dropout,
#                                                     batch_first=True)
#         self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
#         self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

#         # Hierarchical convolutions.
#         self.hierarchical_conv_personality = nn.Sequential(
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU()
#         )
#         self.hierarchical_conv_emotion = nn.Sequential(
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU()
#         )
#         if stim_emo_dim == 0:
#             self.fixed_T_emo = 8   # new value when stim_emo is disabled
#         else:
#             self.fixed_T_emo = 10  # default value when stim_emo is used
#         self.fuse_fc_emo = nn.Sequential(
#             nn.Linear(self.d_model * self.fixed_T_emo, self.d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout_emotion)
#         )

#         self.d_personlity = self.d_model 
#         self.fuse_fc_per = nn.Sequential(
#             nn.Linear(self.d_model, self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person)
#         )

#         # --- Task Attention for Personality vs. Emotion ---
#         self.task_attention = TaskAttentionTemporal(d_model=self.d_model)
#         self.personality_attention_proj = nn.Linear(self.d_model, self.d_personlity)

#         # Trial-level features.
#         self.features_dim = self.d_model//2
#         self.eye_feat_mlp = nn.Sequential(
#             nn.Linear(eye_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.au_feat_mlp = nn.Sequential(
#             nn.Linear(au_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.shimmer_feat_mlp = nn.Sequential(
#             nn.Linear(shimmer_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.features_mlps = nn.Sequential(
#             nn.Linear(self.features_dim * 3, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.fuse_personality = nn.Linear(2 * self.d_model, self.d_model)
#         self.personality_token_proj = nn.Linear(1, self.d_model)

#         # Personality prediction.
#         # Update emo_per_fc input to account for additional attention signal.
#         self.emo_per_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + self.d_model, self.d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout_emotion),
#             nn.Linear(self.d_model, self.d_model)
#         )
#         self.trial_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, personality_dim)
#         )
#         self.rawpers_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, personality_dim)
#         )
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(self.d_personlity + personality_dim, self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person)
#         )

#         # --- Optional Gender Branch ---
#         if self.use_gender:
#             # Input dimension: same as concat_in used for personality prediction.
#             gender_input_dim = self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0)
#             gender_hidden_dim = self.d_model//8  # you can tune this value
#             self.gender_branch = nn.Sequential(
#                 nn.Linear(gender_input_dim, gender_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(gender_hidden_dim, 2)
#             )
#         else:
#             self.gender_branch = None

#         # Emotion heads.
#         self.emotion_heads = nn.ModuleList([
#             AdvancedEmotionHead(self.d_model + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), personality_dim, self.d_model,
#                                 emotion_num_classes, dropout_rate=dropout_emotion)
#             for _ in range(self.emotion_num_labels)
#         ])

#     def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in,
#                 eye_feat, au_feat, shimmer_feat):
#         # --- Use or ignore stim_emo based on flag ---
#         if not self.use_stim_emo:
#             stim_emo = None

#         # --- Process modalities ---
#         # Eye modality
#         eye_emb = self.eye_proj(eye)
#         eye_emb = self.pos_encoding_modality(eye_emb)
#         eye_out = self.eye_transformer(eye_emb)
#         eye_tok = self.project_modality(downsample_sequence_deterministic(eye_out, self.target_length))
#         # Pupil modality
#         pupil_emb = self.pupil_proj(pupil)
#         pupil_emb = self.pos_encoding_modality(pupil_emb)
#         pupil_out = self.pupil_transformer(pupil_emb)
#         pupil_tok = self.project_modality(downsample_sequence_deterministic(pupil_out, self.target_length))
#         # AU modality
#         au_emb = self.au_proj(au)
#         au_emb = self.pos_encoding_modality(au_emb)
#         au_out = self.au_transformer(au_emb)
#         au_tok = self.project_modality(downsample_sequence_deterministic(au_out, self.target_length))
#         # GSR modality
#         gsr_emb = self.gsr_proj(gsr)
#         gsr_emb = self.pos_encoding_modality(gsr_emb)
#         gsr_out = self.gsr_transformer(gsr_emb)
#         gsr_tok = self.project_modality(downsample_sequence_deterministic(gsr_out, self.target_length))
        
#         raw_mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)  # [B, T, d_model]
#         raw_mod_tokens = raw_mod_tokens + self.pos_embedding
#         # --- Project personality prediction into tokens ---
#         # personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
#         # scale_factor = 2.0
#         # personality_tokens = scale_factor * personality_tokens
#         # replication_factor = 3
#         # personality_tokens = personality_tokens.repeat(1, replication_factor, 1)
#         # fused_tokens_with_personality = torch.cat([personality_tokens, fused_tokens], dim=1)
#         mod_tokens = self.modal_transformer(raw_mod_tokens)
#         # print("mod_tokens.shape",mod_tokens.shape)
#         # --- Personality branch ---
#         rep_person, rep_emotion = self.task_attention(mod_tokens)  # each: [B, d_model]
#         # print("rep_person.shape",rep_person.shape)
#         # print("rep_emotion.shape",rep_emotion.shape)
#         rep_person_proj = self.personality_attention_proj(rep_person)  # [B, d_personlity]
#         # print("rep_person_proj.shape",rep_person_proj.shape)
#         x = rep_person_proj.transpose(1, 2)  # [B, d_model, T]
#         # print("x.shape",x.shape)
#         x_per = self.hierarchical_conv_personality(x)  # [B, d_model, T_per]
#         fused_per = self.fuse_fc_per(x_per.mean(dim=2))  # [B, d_personlity]
        
#         # x_per_upsampled = F.interpolate(x_per, size=raw_mod_tokens.size(1), mode='linear', align_corners=False)
#         # x_per_upsampled = x_per_upsampled.transpose(1, 2)  # [B, T, d_model]
#         # if not self.use_personality:
#         #     x_per_upsampled = torch.zeros_like(x_per_upsampled)
#         # fused_tokens = torch.cat([raw_mod_tokens, x_per_upsampled], dim=-1)  # [B, T, 2*d_model]
#         # fused_tokens = self.fuse_personality(fused_tokens)  # [B, T, d_model]
        
#         # --- Task Attention ---
        
        
#         # --- Trial-level features & Personality Prediction ---
#         eye_trial = self.eye_feat_mlp(eye_feat)
#         au_trial = self.au_feat_mlp(au_feat)
#         shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
#         trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
#         trial_features = self.features_mlps(trial_concat)
#         if self.use_personality:
#             if self.use_stim_emo:
#                 # Include stim_emo: expected total dims = 64+32+6 = 102.
#                 concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
#             else:
#                 # Exclude stim_emo: expected total dims = 64+32 = 96.
#                 concat_in = torch.cat([fused_per, trial_features], dim=1)
#             pred_personality = self.rawpers_fc(concat_in)
#             if self.finetune_mode:
#                 trial_delta = self.trial_fc(concat_in)
#                 trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
#             else:
#                 trial_delta = torch.zeros_like(pred_personality)
#             golden_personality = pred_personality + trial_delta
#         else:
#             # If personality is not used, return zeros.
#             batch_size = fused_per.size(0)
#             pred_personality = torch.zeros(batch_size, self.personality_dim, device=fused_per.device)
#             trial_delta = torch.zeros_like(pred_personality)
#             golden_personality = pred_personality

#         # --- Optional Gender Prediction ---
#         gender_pred = None
#         if self.use_gender:
#             gender_pred = self.gender_branch(concat_in)
        
#         # # --- Project personality prediction into tokens ---
#         # personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
#         # scale_factor = 2.0
#         # personality_tokens = scale_factor * personality_tokens
#         # replication_factor = 3
#         # personality_tokens = personality_tokens.repeat(1, replication_factor, 1)
#         # fused_tokens_with_personality = torch.cat([personality_tokens, fused_tokens], dim=1)
#         # mod_tokens = self.modal_transformer(fused_tokens_with_personality)
        
#         # --- Emotion branch ---
#         x_emo = self.hierarchical_conv_emotion(rep_emotion.transpose(1, 2))
#         # Use CPU workaround for adaptive pooling on MPS.
#         x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)
#         x_emo_flat = x_emo_pooled.flatten(start_dim=1)
#         fused_emo = self.fuse_fc_emo(x_emo_flat)
#         # print("fused_emo.shape",fused_emo.shape)
#         # print("fused_per.shape",fused_per.shape)
#         fused_emo = self.emo_per_fc(torch.cat([fused_per, fused_emo], dim=1))
        
#         if stim_emo is not None:
#             concat_in_emo = torch.cat([fused_emo, trial_features, stim_emo], dim=1)
#         else:
#             concat_in_emo = torch.cat([fused_emo, trial_features], dim=1)
#         emotion_logits_list = []
#         for head in self.emotion_heads:
#             logits = head(concat_in_emo, golden_personality)
#             emotion_logits_list.append(logits.unsqueeze(1))
#         emotion_logits = torch.cat(emotion_logits_list, dim=1)
        
#         return trial_delta, pred_personality, emotion_logits, fused_per, gender_pred
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedConcatFusionMultiModalModelTransformerAttention_v2(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.4, dropout_emotion=0.25, dropout_features=0.3,
                 embedding_noise_std=0.0, max_val=1.0,
                 emotion_num_labels=4,
                 use_gender=False,        
                 use_stim_emo=True,
                 use_personality=True):
        super(AdvancedConcatFusionMultiModalModelTransformerAttention_v2, self).__init__()
        # Cast to int.
        hidden_dim = int(hidden_dim)
        proj_dim = int(proj_dim)
        target_length = int(target_length)
        personality_dim = int(personality_dim)
        emotion_num_classes = int(emotion_num_classes)
        transformer_nhead = int(transformer_nhead)
        transformer_layers = int(transformer_layers)

        self.stim_emo_dim = stim_emo_dim
        self.embedding_noise_std = embedding_noise_std
        self.personality_dim = personality_dim
        self.max_val = max_val
        self.target_length = target_length
        self.finetune_mode = False
        self.emotion_num_labels = emotion_num_labels
        self.use_gender = use_gender
        self.use_stim_emo = use_stim_emo
        self.use_personality = use_personality

        # Define dimension for small modality transformers.
        self.transformer_small_dim = 64  
        self.transformer_small_head = 2

        # Project raw inputs.
        self.eye_proj    = nn.Linear(eye_dim, self.transformer_small_dim)
        self.pupil_proj  = nn.Linear(pupil_dim, self.transformer_small_dim)
        self.au_proj     = nn.Linear(au_dim, self.transformer_small_dim)
        self.gsr_proj    = nn.Linear(gsr_dim, self.transformer_small_dim)

        self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)
        transformer_dropout = 0.25

        # Modality-specific transformers.
        self.eye_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim,
                                       nhead=self.transformer_small_head,
                                       dropout=transformer_dropout,
                                       batch_first=True),
            num_layers=1
        )
        self.pupil_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim,
                                       nhead=self.transformer_small_head,
                                       dropout=transformer_dropout,
                                       batch_first=True),
            num_layers=1
        )
        self.au_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim,
                                       nhead=self.transformer_small_head,
                                       dropout=transformer_dropout,
                                       batch_first=True),
            num_layers=1
        )
        self.gsr_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim,
                                       nhead=self.transformer_small_head,
                                       dropout=transformer_dropout,
                                       batch_first=True),
            num_layers=1
        )
        self.project_modality = nn.Linear(self.transformer_small_dim, proj_dim)

        # Transformer for modality fusion.
        self.d_model = proj_dim * 4  # final dimension after cat(4 tokens)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=transformer_nhead,
                                                   dropout=transformer_dropout,
                                                   batch_first=True)
        self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

        # Hierarchical convolutions.
        self.hierarchical_conv_personality = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        # Decide how many time steps remain after the conv stack
        if stim_emo_dim == 0:
            self.fixed_T_emo = 8   # new value when stim_emo is disabled
        else:
            self.fixed_T_emo = 10  # default value when stim_emo is used

        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(self.d_model * self.fixed_T_emo, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )

        self.d_personlity = self.d_model
        self.fuse_fc_per = nn.Sequential(
            nn.Linear(self.d_model, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # --- Task Attention for Personality vs. Emotion ---
        self.task_attention = TaskAttentionTemporal(d_model=self.d_model)
        self.personality_attention_proj = nn.Linear(self.d_model, self.d_personlity)

        # Trial-level features.
        self.features_dim = self.d_model // 2
        self.eye_feat_mlp = nn.Sequential(
            nn.Linear(eye_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.au_feat_mlp = nn.Sequential(
            nn.Linear(au_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.shimmer_feat_mlp = nn.Sequential(
            nn.Linear(shimmer_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.features_mlps = nn.Sequential(
            nn.Linear(self.features_dim * 3, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.fuse_personality = nn.Linear(2 * self.d_model, self.d_model)
        self.personality_token_proj = nn.Linear(1, self.d_model)

        # Personality prediction.
        self.emo_per_fc = nn.Sequential(
            nn.Linear(self.d_personlity + self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion),
            nn.Linear(self.d_model, self.d_model)
        )
        self.trial_fc = nn.Sequential(
            nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0),
                      self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.rawpers_fc = nn.Sequential(
            nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0),
                      self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_personlity + personality_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # --- Optional Gender Branch ---
        if self.use_gender:
            gender_input_dim = self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0)
            gender_hidden_dim = self.d_model // 8
            self.gender_branch = nn.Sequential(
                nn.Linear(gender_input_dim, gender_hidden_dim),
                nn.ReLU(),
                nn.Linear(gender_hidden_dim, 2)
            )
        else:
            self.gender_branch = None

        # Emotion heads.
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead_v2(
                # input_dim = d_model (for fused_emo) + features_dim + possible stim_emo
                self.d_model + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0),
                # persona_dim
                personality_dim,
                self.stim_emo_dim if self.use_stim_emo else 0,
                # hidden_dim
                self.d_model,
                emotion_num_classes,
                dropout_rate=dropout_emotion,
            )
            for _ in range(self.emotion_num_labels)
        ])

        ## SKIP CONNECTION MLP for Emotion Branch:
        # We'll average (eye_tok + pupil_tok + au_tok + gsr_tok) / 4 => [B,T,proj_dim],
        # then flatten to [B, T*proj_dim], pass it through a small MLP => [B, d_model].
        self.skip_emo_mlp = nn.Sequential(
            nn.Linear(target_length * proj_dim*4, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in,
                eye_feat, au_feat, shimmer_feat):
        # --- Use or ignore stim_emo based on flag ---
        if not self.use_stim_emo:
            stim_emo = None

        # --- Process modalities ---
        # Eye
        eye_emb = self.eye_proj(eye)                       # [B, seq_len, 64]
        eye_emb = self.pos_encoding_modality(eye_emb)
        eye_out = self.eye_transformer(eye_emb)            # [B, seq_len, 64]
        eye_tok = self.project_modality(
            downsample_sequence_deterministic(eye_out, self.target_length)
        )  # [B, T, proj_dim]

        # Pupil
        pupil_emb = self.pupil_proj(pupil)
        pupil_emb = self.pos_encoding_modality(pupil_emb)
        pupil_out = self.pupil_transformer(pupil_emb)
        pupil_tok = self.project_modality(
            downsample_sequence_deterministic(pupil_out, self.target_length)
        )  # [B, T, proj_dim]

        # AU
        au_emb = self.au_proj(au)
        au_emb = self.pos_encoding_modality(au_emb)
        au_out = self.au_transformer(au_emb)
        au_tok = self.project_modality(
            downsample_sequence_deterministic(au_out, self.target_length)
        )  # [B, T, proj_dim]

        # GSR
        gsr_emb = self.gsr_proj(gsr)
        gsr_emb = self.pos_encoding_modality(gsr_emb)
        gsr_out = self.gsr_transformer(gsr_emb)
        gsr_tok = self.project_modality(
            downsample_sequence_deterministic(gsr_out, self.target_length)
        )  # [B, T, proj_dim]

        # Combine these 4 tokens along -1 => shape [B, T, proj_dim*4] = [B, T, self.d_model]
        raw_mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)
        raw_mod_tokens = raw_mod_tokens + self.pos_embedding  # [B, T, d_model]

        # Pass them through the "modal" transformer
        mod_tokens = self.modal_transformer(raw_mod_tokens)  # [B, T, d_model]

        # --- Task Attention ---
        rep_person, rep_emotion = self.task_attention(mod_tokens)  # each: [B, d_model]
        rep_person_proj = self.personality_attention_proj(rep_person)  # [B, d_personlity]

        # Personality path
        x = rep_person_proj.transpose(1, 2)        # => [B, d_model, 1] but you do further conv with T?
        x_per = self.hierarchical_conv_personality(x)  # => [B, d_model, T_per]
        fused_per = self.fuse_fc_per(x_per.mean(dim=2))  # => [B, d_personlity]

        # --- Trial-level features ---
        eye_trial = self.eye_feat_mlp(eye_feat)
        au_trial = self.au_feat_mlp(au_feat)
        shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
        trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
        trial_features = self.features_mlps(trial_concat)

        # Personality prediction
        if self.use_personality:
            if self.use_stim_emo:
                # shape: [B, d_personlity + features_dim + stim_emo_dim]
                concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
            else:
                concat_in = torch.cat([fused_per, trial_features], dim=1)

            pred_personality = self.rawpers_fc(concat_in)
            if self.finetune_mode:
                trial_delta = self.trial_fc(concat_in)
                trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
            else:
                trial_delta = torch.zeros_like(pred_personality)
            golden_personality = pred_personality + trial_delta
        else:
            # If personality is not used, return zeros
            batch_size = fused_per.size(0)
            pred_personality = torch.zeros(batch_size, self.personality_dim, device=fused_per.device)
            trial_delta = torch.zeros_like(pred_personality)
            golden_personality = pred_personality

        # --- Optional Gender Prediction ---
        gender_pred = None
        if self.use_gender:
            gender_pred = self.gender_branch(concat_in)

        # --- Emotion branch ---
        # 1) Convolution-based path
        x_emo = self.hierarchical_conv_emotion(rep_emotion.transpose(1, 2))  # [B, d_model, T_emo]
        x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)
        x_emo_flat = x_emo_pooled.flatten(start_dim=1)     # [B, d_model*T_emo]
        fused_emo = self.fuse_fc_emo(x_emo_flat)           # => [B, d_model]

        # 2) ## SKIP CONNECTION: incorporate early features ##
        # Average the 4 tokens => shape [B, T, proj_dim], flatten => [B, T*proj_dim]
        # skip_tokens = (eye_tok + pupil_tok + au_tok + gsr_tok) / 4.0  # [B, T, proj_dim]
        skip_tokens=rep_emotion
        B, T, _ = skip_tokens.shape
        skip_tokens_flat = skip_tokens.view(B, T * self.personality_attention_proj.out_features)
        skip_early_emo = self.skip_emo_mlp(skip_tokens_flat)  # => [B, d_model]

        # Merge skip_early_emo with the final conv-based emotion vector
        fused_emo_skip = fused_emo + skip_early_emo  # shape [B, d_model]

        # 3) Combine with personality for final fusion
        # fused_emo_skip = self.emo_per_fc(torch.cat([fused_per, fused_emo_skip], dim=1))  # => [B, d_model]

        # 4) Build input for emotion heads
        if stim_emo is not None:
            concat_in_emo = torch.cat([fused_emo_skip, trial_features, stim_emo], dim=1)
        else:
            concat_in_emo = torch.cat([fused_emo_skip, trial_features], dim=1)

        emotion_logits_list = []
        for head in self.emotion_heads:
            if self.use_stim_emo:
                logits = head(concat_in_emo, fused_per,stim_emo)
            else:

                logits = head(concat_in_emo, fused_per)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)

        return trial_delta, pred_personality, emotion_logits, fused_per, gender_pred

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradReverseLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradReverseLayer, self).__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

class AdvancedConcatFusionMultiModalModelTransformerAttentionUserAdversold(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
                 embedding_noise_std=0.0, max_val=1.0,
                 emotion_num_labels=4,
                 use_gender=False,        # NEW: flag for gender branch
                 use_stim_emo=True,use_personality=True):      # NEW: flag to use stim_emo input
        super(AdvancedConcatFusionMultiModalModelTransformerAttentionUserAdversold, self).__init__()
        # Cast to int.
        hidden_dim = int(hidden_dim)
        proj_dim = int(proj_dim)
        target_length = int(target_length)
        personality_dim = int(personality_dim)
        emotion_num_classes = int(emotion_num_classes)
        transformer_nhead = int(transformer_nhead)
        transformer_layers = int(transformer_layers)
        self.stim_emo_dim = stim_emo_dim
        self.embedding_noise_std = embedding_noise_std
        self.personality_dim = personality_dim
        self.max_val = max_val
        self.target_length = target_length
        self.finetune_mode = False
        self.emotion_num_labels = emotion_num_labels
        self.use_gender = use_gender          # NEW flag
        self.use_stim_emo = use_stim_emo      # NEW flag
        self.use_personality = use_personality # NEW flag
        # Define dimension for small modality transformers.
        self.transformer_small_dim = 64  
        self.transformer_small_head = 2

        # Project raw inputs.
        self.eye_proj    = nn.Linear(eye_dim, self.transformer_small_dim)
        self.pupil_proj  = nn.Linear(pupil_dim, self.transformer_small_dim)
        self.au_proj     = nn.Linear(au_dim, self.transformer_small_dim)
        self.gsr_proj    = nn.Linear(gsr_dim, self.transformer_small_dim)
        
        self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)
        
        # Modality-specific transformers.
        self.eye_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.3, batch_first=True),
            num_layers=1
        )
        self.pupil_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.3, batch_first=True),
            num_layers=1
        )
        self.au_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.3, batch_first=True),
            num_layers=1
        )
        self.gsr_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
                                       dropout=0.3, batch_first=True),
            num_layers=1
        )
        self.project_modality = nn.Linear(self.transformer_small_dim, proj_dim)

        # Transformer for modality fusion.
        self.d_model = proj_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                    nhead=transformer_nhead,
                                                    dropout=0.3,
                                                    batch_first=True)
        self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

        # Hierarchical convolutions.
        self.hierarchical_conv_personality = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
                nn.LayerNorm([self.d_model, self.target_length]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
                nn.LayerNorm([self.d_model, self.target_length // 2]),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        if stim_emo_dim == 0:
            self.fixed_T_emo = 8   # new value when stim_emo is disabled
        else:
            self.fixed_T_emo = 10  # default value when stim_emo is used
        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(self.d_model * self.fixed_T_emo, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )

        self.d_personlity = self.d_model // 2
        self.fuse_fc_per = nn.Sequential(
            nn.Linear(self.d_model, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # --- Task Attention for Personality vs. Emotion ---
        self.task_attention = TaskAttention(d_model=self.d_model)
        self.personality_attention_proj = nn.Linear(self.d_model, self.d_personlity)

        # Trial-level features.
        self.features_dim = 32
        self.eye_feat_mlp = nn.Sequential(
            nn.Linear(eye_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.au_feat_mlp = nn.Sequential(
            nn.Linear(au_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.shimmer_feat_mlp = nn.Sequential(
            nn.Linear(shimmer_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.features_mlps = nn.Sequential(
            nn.Linear(self.features_dim * 3, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.fuse_personality = nn.Linear(2 * self.d_model, self.d_model)
        self.personality_token_proj = nn.Linear(1, self.d_model)

        # Personality prediction.
        # Update emo_per_fc input to account for additional attention signal.
        self.emo_per_fc = nn.Sequential(
            nn.Linear(self.d_personlity + 2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion),
            nn.Linear(self.d_model, self.d_model)
        )
        self.trial_fc = nn.Sequential(
            nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.rawpers_fc = nn.Sequential(
            nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personlity, personality_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_personlity + personality_dim, self.d_personlity),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # SUGGESTION 4: We'll pass user_dim to all heads that want user embeddings
        # But we keep the old "user_classifier" for adversarial ID classification
        self.grad_reverse = GradReverseLayer(lambda_=1.0)
        self.user_classifier = nn.Sequential(
            nn.Linear(self.d_model + self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_users)
        )

        # --- Optional Gender Branch ---
        if self.use_gender:
            # Input dimension: same as concat_in used for personality prediction.
            gender_input_dim = self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0)
            gender_hidden_dim = 32  # you can tune this value
            self.gender_branch = nn.Sequential(
                nn.Linear(gender_input_dim, gender_hidden_dim),
                nn.ReLU(),
                nn.Linear(gender_hidden_dim, 2)
            )
        else:
            self.gender_branch = None

        # Emotion heads.
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead(self.d_model + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), personality_dim, self.d_model,
                                emotion_num_classes, dropout_rate=dropout_emotion)
            for _ in range(self.emotion_num_labels)
        ])

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in,
                eye_feat, au_feat, shimmer_feat):
        # --- Use or ignore stim_emo based on flag ---
        if not self.use_stim_emo:
            stim_emo = None

        # --- Process modalities ---
        # Eye modality
        eye_emb = self.eye_proj(eye)
        eye_emb = self.pos_encoding_modality(eye_emb)
        eye_out = self.eye_transformer(eye_emb)
        eye_tok = self.project_modality(downsample_sequence_deterministic(eye_out, self.target_length))
        # Pupil modality
        pupil_emb = self.pupil_proj(pupil)
        pupil_emb = self.pos_encoding_modality(pupil_emb)
        pupil_out = self.pupil_transformer(pupil_emb)
        pupil_tok = self.project_modality(downsample_sequence_deterministic(pupil_out, self.target_length))
        # AU modality
        au_emb = self.au_proj(au)
        au_emb = self.pos_encoding_modality(au_emb)
        au_out = self.au_transformer(au_emb)
        au_tok = self.project_modality(downsample_sequence_deterministic(au_out, self.target_length))
        # GSR modality
        gsr_emb = self.gsr_proj(gsr)
        gsr_emb = self.pos_encoding_modality(gsr_emb)
        gsr_out = self.gsr_transformer(gsr_emb)
        gsr_tok = self.project_modality(downsample_sequence_deterministic(gsr_out, self.target_length))
        
        raw_mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)  # [B, T, d_model]
        raw_mod_tokens = raw_mod_tokens + self.pos_embedding
        pooled_raw = raw_mod_tokens.mean(dim=1)  # [B, self.d_model]

        # --- Personality branch ---
        x = raw_mod_tokens.transpose(1, 2)  # [B, d_model, T]
        x_per = self.hierarchical_conv_personality(x)  # [B, d_model, T_per]
        fused_per = self.fuse_fc_per(x_per.mean(dim=2))  # [B, d_personlity]
        
        x_per_upsampled = F.interpolate(x_per, size=raw_mod_tokens.size(1), mode='linear', align_corners=False)
        x_per_upsampled = x_per_upsampled.transpose(1, 2)  # [B, T, d_model]
        if not self.use_personality:
            x_per_upsampled = torch.zeros_like(x_per_upsampled)
        fused_tokens = torch.cat([raw_mod_tokens, x_per_upsampled], dim=-1)  # [B, T, 2*d_model]
        fused_tokens = self.fuse_personality(fused_tokens)  # [B, T, d_model]
        
        # --- Task Attention ---
        rep_person, rep_emotion = self.task_attention(fused_tokens)  # each: [B, d_model]
        rep_person_proj = self.personality_attention_proj(rep_person)  # [B, d_personlity]
        final_person_rep = fused_per + rep_person_proj  # [B, d_personlity]
        
        # --- Trial-level features & Personality Prediction ---
        eye_trial = self.eye_feat_mlp(eye_feat)
        au_trial = self.au_feat_mlp(au_feat)
        shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
        trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
        trial_features = self.features_mlps(trial_concat)

# --- (4) Combine pooled raw_mod_tokens w/ trial_features for user classifier ---
        user_adv_in = torch.cat([pooled_raw, trial_features], dim=1)  # shape [B, d_model + features_dim]

        # If you want, reduce dimension or directly pass to grad reversal:
        rev_user = self.grad_reverse(user_adv_in)
        user_logits = self.user_classifier(rev_user)  # shape [B, num_users]

        if self.use_personality:
            if self.use_stim_emo:
                # Include stim_emo: expected total dims = 64+32+6 = 102.
                concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
            else:
                # Exclude stim_emo: expected total dims = 64+32 = 96.
                concat_in = torch.cat([fused_per, trial_features], dim=1)
            pred_personality = self.rawpers_fc(concat_in)
            if self.finetune_mode:
                trial_delta = self.trial_fc(concat_in)
                trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
            else:
                trial_delta = torch.zeros_like(pred_personality)
            golden_personality = pred_personality + trial_delta
        else:
            # If personality is not used, return zeros.
            batch_size = fused_per.size(0)
            pred_personality = torch.zeros(batch_size, self.personality_dim, device=fused_per.device)
            trial_delta = torch.zeros_like(pred_personality)
            golden_personality = pred_personality

        # --- Optional Gender Prediction ---
        gender_pred = None
        if self.use_gender:
            gender_pred = self.gender_branch(concat_in)
        
        # --- Project personality prediction into tokens ---
        personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
        scale_factor = 2.0
        personality_tokens = scale_factor * personality_tokens
        replication_factor = 3
        personality_tokens = personality_tokens.repeat(1, replication_factor, 1)
        fused_tokens_with_personality = torch.cat([personality_tokens, fused_tokens], dim=1)
        mod_tokens = self.modal_transformer(fused_tokens_with_personality)
        
        # --- Emotion branch ---
        x_emo = self.hierarchical_conv_emotion(mod_tokens.transpose(1, 2))
        # Use CPU workaround for adaptive pooling on MPS.
        x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)
        x_emo_flat = x_emo_pooled.flatten(start_dim=1)
        fused_emo = self.fuse_fc_emo(x_emo_flat)
        fused_emo = self.emo_per_fc(torch.cat([final_person_rep, fused_emo, rep_emotion], dim=1))
        
        if stim_emo is not None:
            concat_in_emo = torch.cat([fused_emo, trial_features, stim_emo], dim=1)
        else:
            concat_in_emo = torch.cat([fused_emo, trial_features], dim=1)
        emotion_logits_list = []
        for head in self.emotion_heads:
            logits = head(concat_in_emo, golden_personality)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)
        
        return trial_delta, pred_personality, emotion_logits, fused_per, gender_pred,user_logits
       
def scale_with_sigmoid(x, max_val):
    return (torch.sigmoid(x) - 0.5) * 0.1 * max_val

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Sparsemax implementation
# ------------------------------
def sparsemax(input, dim=-1):
    """
    Sparsemax: a sparse alternative to softmax.
    Implementation adapted from:
    "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
    """
    original_size = input.size()
    input = input.transpose(dim, -1)
    input = input.reshape(-1, input.size(-1))
    
    # sort input in descending order
    zs, _ = torch.sort(input, dim=1, descending=True)
    range_vals = torch.arange(1, input.size(1) + 1, device=input.device, dtype=input.dtype).unsqueeze(0)
    zs_cumsum = zs.cumsum(dim=1)
    bound = 1 + range_vals * zs
    is_gt = (bound > zs_cumsum).to(input.dtype)
    k = is_gt.sum(dim=1, keepdim=True)
    zs_sparse = torch.gather(zs, 1, k.long() - 1)
    tau = (torch.gather(zs_cumsum, 1, k.long() - 1) - 1) / k
    output = torch.clamp(input - tau, min=0)
    output = output.reshape(original_size)
    return output

# ------------------------------
# GradReverse for adversarial loss
# ------------------------------
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradReverseLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradReverseLayer, self).__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

# ------------------------------
# Main Model
# ------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Define the Emotion Signal Attention Module
# ------------------------------
class EmotionSignalAttention(nn.Module):
    """
    This module takes a list of input tokens (features) and computes a weighted sum
    by projecting each token to a common dimension and learning an attention score.
    
    Args:
        input_dims (list[int]): List of input dimensions for each token.
        common_dim (int): Dimension into which each token is projected.
        hidden_dim (int): Hidden dimension for computing attention scores.
    """
    def __init__(self, input_dims, common_dim, hidden_dim):
        super().__init__()
        self.num_tokens = len(input_dims)
        # One linear projection per token
        self.projections = nn.ModuleList([
            nn.Linear(dim, common_dim) for dim in input_dims
        ])
        # Attention scoring MLP
        self.attn_fc = nn.Linear(common_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
    
    def forward(self, tokens):
        """
        Args:
            tokens (list[torch.Tensor]): Each tensor is of shape [B, input_dim_i].
        Returns:
            weighted_sum (torch.Tensor): Combined representation of shape [B, common_dim].
            attn_weights (torch.Tensor): Attention weights for each token, shape [B, num_tokens].
        """
        projected_tokens = []
        for proj, token in zip(self.projections, tokens):
            projected_tokens.append(proj(token))  # [B, common_dim]
        # Stack along a new dimension: shape [B, num_tokens, common_dim]
        tokens_stack = torch.stack(projected_tokens, dim=1)
        # Compute attention scores per token
        hidden = torch.tanh(self.attn_fc(tokens_stack))  # [B, num_tokens, hidden_dim]
        scores = self.attn_score(hidden)  # [B, num_tokens, 1]
        attn_weights = F.softmax(scores, dim=1)  # [B, num_tokens, 1]
        # Weighted sum over tokens
        weighted_sum = torch.sum(tokens_stack * attn_weights, dim=1)  # [B, common_dim]
        return weighted_sum, attn_weights.squeeze(-1)

# ------------------------------
# Your Updated Model
# ------------------------------
class AdvancedConcatFusionMultiModalModelTransformerAttentionUserAdvers(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
                 embedding_noise_std=0.0, max_val=1.0,
                 emotion_num_labels=4,
                 use_gender=False,        # flag for gender branch
                 use_stim_emo=True, 
                 use_personality=True,
                 sparse_attn_coef=0.01,
                 base_dict_size=10,      # universal dictionary size
                 ext_dict_size=10):      # extended dictionary size
        super().__init__()
        # Cast hyperparameters to int
        hidden_dim = int(hidden_dim)
        proj_dim = int(proj_dim)
        target_length = int(target_length)
        personality_dim = int(personality_dim)
        emotion_num_classes = int(emotion_num_classes)
        transformer_nhead = int(transformer_nhead)
        transformer_layers = int(transformer_layers)
        
        self.stim_emo_dim = stim_emo_dim
        self.embedding_noise_std = embedding_noise_std
        self.personality_dim = personality_dim
        self.max_val = max_val
        self.target_length = target_length
        self.finetune_mode = False
        self.emotion_num_labels = emotion_num_labels
        self.use_gender = use_gender
        self.use_stim_emo = use_stim_emo
        self.use_personality = use_personality
        self.sparse_attn_coef = sparse_attn_coef
        self.d_model = proj_dim * 4
        self.d_personality = self.d_model // 2
        # Additional hyperparams for controlling losses
        self.lambda_consistency = 0.0
        self.user_loss_weight = 0.33

        # Transformer config for smaller modality branches
        self.transformer_small_dim = 64  
        self.transformer_small_head = 2
        self.features_dim = 32
        # ------------------------------
        # 1) Universal + Extended Dictionary
        # ------------------------------
        # Define dictionary dimensions.
        # In __init__:
        self.dict_dim = 256
        self.user_embedding_dim = self.dict_dim  # final user embedding dimension

        # Define an initial dictionary: one vector per user.
        self.initial_user_dict = nn.Parameter(torch.randn(num_users, self.dict_dim))
        # We'll derive a universal dictionary with k entries via PCA (SVD) from the rows corresponding to users in a batch.
        self.k_universal = 10

        # Define a dictionary modulation network:
        self.dict_mlp = nn.Sequential(
            nn.Linear(self.d_personality + self.features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.dict_dim)
        )

        # Update the user selector to have output dimension k_universal + 1 (for the personal vector)
        self.user_selector = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.k_universal + 1)
        )


# (Remove extended_dict and ext_dict_transform if they are no longer needed.)

        # ------------------------------
        # 2) Modality-Specific Projects & Transformers
        # ------------------------------
        self.eye_proj    = nn.Linear(eye_dim, self.transformer_small_dim)
        self.pupil_proj  = nn.Linear(pupil_dim, self.transformer_small_dim)
        self.au_proj     = nn.Linear(au_dim, self.transformer_small_dim)
        self.gsr_proj    = nn.Linear(gsr_dim, self.transformer_small_dim)
        self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)

        def make_small_transformer(d_model, nhead):
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dropout=0.3, batch_first=True),
                num_layers=1
            )

        self.eye_transformer    = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.pupil_transformer  = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.au_transformer     = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.gsr_transformer    = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        
        self.project_modality   = nn.Linear(self.transformer_small_dim, proj_dim)

        # ------------------------------
        # 3) Transformer for Modality Fusion
        # ------------------------------
          # e.g. 128 if proj_dim=32
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                    nhead=transformer_nhead,
                                                    dropout=0.3,
                                                    batch_first=True)
        self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

        # ------------------------------
        # 4) Hierarchical Convolutions (Personality vs. Emotion)
        # ------------------------------
        self.hierarchical_conv_personality = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([self.d_model, target_length]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([self.d_model, target_length // 2]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )

        if stim_emo_dim == 0:
            self.fixed_T_emo = 8
        else:
            self.fixed_T_emo = 10

        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(self.d_model * self.fixed_T_emo, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )

        # ------------------------------
        # 5) Personality Branch & Task Attention
        # ------------------------------
        
        self.task_attention = TaskAttention(d_model=self.d_model)
        self.personality_attention_proj = nn.Linear(self.d_model, self.d_personality)

        base_personality_in_dim = self.user_embedding_dim + (stim_emo_dim if self.use_stim_emo else 0)
        self.fuse_fc_per = nn.Sequential(
            nn.Linear(self.d_model, self.d_personality),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # ------------------------------
        # 6) Trial-Level Features
        # ------------------------------
        
        self.eye_feat_mlp = nn.Sequential(
            nn.Linear(eye_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.au_feat_mlp = nn.Sequential(
            nn.Linear(au_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.shimmer_feat_mlp = nn.Sequential(
            nn.Linear(shimmer_feat_dim, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.features_mlps = nn.Sequential(
            nn.Linear(self.features_dim * 3, self.features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_features)
        )
        self.fuse_personality = nn.Linear(2 * self.d_model, self.d_model)
        self.personality_token_proj = nn.Linear(1, self.d_model)

        self.emotion_to_personality = nn.Sequential(
            nn.Linear(self.d_model, personality_dim)
        )

        # ------------------------------
        # 7) Personality Prediction Heads
        # ------------------------------
        self.rawpers_fc = nn.Sequential(
            nn.Linear(base_personality_in_dim, self.d_personality),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personality, self.d_personality),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(self.d_personality, personality_dim)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_personality + personality_dim, self.d_personality),
            nn.ReLU(),
            nn.Dropout(dropout_person)
        )

        # ------------------------------
        # 8) Adversarial User Classifier
        # ------------------------------
        self.grad_reverse = GradReverseLayer(lambda_=1.0)
        self.user_classifier = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )
        self.user_classifier_before = nn.Sequential(
            nn.Linear(self.d_personality, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )
        self.user_classifier_after = nn.Sequential(
            nn.Linear(self.user_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )

        # ------------------------------
        # 9) Optional Gender Branch
        # ------------------------------
        if self.use_gender:
            gender_input_dim = base_personality_in_dim + self.features_dim
            gender_hidden_dim = 32
            self.gender_branch = nn.Sequential(
                nn.Linear(gender_input_dim, gender_hidden_dim),
                nn.ReLU(),
                nn.Linear(gender_hidden_dim, 2)
            )
        else:
            self.gender_branch = None


        
        # ------------------------------
        # 10) Emotion Heads
        # ------------------------------
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead(self.d_model + self.features_dim + 64 + (self.stim_emo_dim if self.use_stim_emo else 0),
                                personality_dim, self.d_model, emotion_num_classes, dropout_rate=dropout_emotion)
            for _ in range(self.emotion_num_labels)
        ])

        self.emo_per_fc = nn.Sequential(
            nn.Linear(self.user_embedding_dim + 2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_emotion),
            nn.Linear(self.d_model, self.d_model)
        )

        # ------------------------------
        # 11) Emotion Signal Attention
        # ------------------------------
        # Define the list of input dimensions for the signals going to the emotion head.
        # Signals: fused_emo_full (from emo_per_fc, output dim = self.d_model),
        # trial_features (output dim = self.features_dim),
        # fused_per (output dim = self.d_personality),
        # and optionally stim_emo (dim = stim_emo_dim).
        if self.use_stim_emo:
            input_dims = [self.d_model, self.features_dim, self.d_personality, self.stim_emo_dim]
            fused_input_dim = self.d_model + self.features_dim + self.d_personality + self.stim_emo_dim
        else:
            input_dims = [self.d_model, self.features_dim, self.d_personality]
            fused_input_dim = self.d_model + self.features_dim + self.d_personality

        # Set hidden_dim for the attention module (e.g., 64)
        self.emotion_attn = EmotionSignalAttention(input_dims, common_dim=fused_input_dim, hidden_dim=64)

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in,
                eye_feat, au_feat, shimmer_feat):
        # 1) Possibly ignore stim_emo if disabled
        if not self.use_stim_emo:
            stim_emo = None

        # 2) Process raw modalities
        eye_emb    = self.eye_proj(eye)
        pupil_emb  = self.pupil_proj(pupil)
        au_emb     = self.au_proj(au)
        gsr_emb    = self.gsr_proj(gsr)

        def encode_and_downsample(x, transformer):
            x = self.pos_encoding_modality(x)
            x = transformer(x)
            x = downsample_sequence_deterministic(x, self.target_length)
            return self.project_modality(x)

        eye_tok   = encode_and_downsample(eye_emb, self.eye_transformer)
        pupil_tok = encode_and_downsample(pupil_emb, self.pupil_transformer)
        au_tok    = encode_and_downsample(au_emb, self.au_transformer)
        gsr_tok   = encode_and_downsample(gsr_emb, self.gsr_transformer)
        
        raw_mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)  # [B, T, d_model]
        raw_mod_tokens = raw_mod_tokens + self.pos_embedding[:, :raw_mod_tokens.size(1), :]
        pooled_raw = raw_mod_tokens.mean(dim=1)  # [B, d_model]

        # 3) Personality Branch
        x_personality = raw_mod_tokens.transpose(1, 2)
        x_per = self.hierarchical_conv_personality(x_personality)
        fused_per = self.fuse_fc_per(x_per.mean(dim=2))  # [B, d_personality]

        x_per_upsampled = F.interpolate(x_per, size=raw_mod_tokens.size(1),
                                        mode='linear', align_corners=False)
        x_per_upsampled = x_per_upsampled.transpose(1, 2)
        if not self.use_personality:
            x_per_upsampled = torch.zeros_like(x_per_upsampled)
        fused_tokens = torch.cat([raw_mod_tokens, x_per_upsampled], dim=-1)
        fused_tokens = self.fuse_personality(fused_tokens)  # [B, T, d_model]

        # Task Attention
        rep_person, rep_emotion = self.task_attention(fused_tokens)
        rep_person_proj = self.personality_attention_proj(rep_person)
        personality_from_emotion = self.emotion_to_personality(rep_emotion)

        # 4) Trial-level features
        eye_trial      = self.eye_feat_mlp(eye_feat)
        au_trial       = self.au_feat_mlp(au_feat)
        shimmer_trial  = self.shimmer_feat_mlp(shimmer_feat)
        trial_concat   = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
        trial_features = self.features_mlps(trial_concat)

        # 5) Build the universal dictionary from the initial dictionary such that
        # --- Dictionary Block ---
        # Gather unique user IDs from the batch.
        unique_ids = torch.unique(user_id)  # shape: [num_unique]
        user_specific_unique = self.initial_user_dict[unique_ids]  # shape: [num_unique, dict_dim]

        # Compute PCA-like universal dictionary via SVD over the unique user-specific rows.
        # --- New Dictionary Block ---
        with torch.no_grad():
            # Compute SVD on the entire initial dictionary to get fixed universal components.
            initial_dict_cpu = self.initial_user_dict.cpu()
            U, S, Vh = torch.linalg.svd(initial_dict_cpu, full_matrices=False)
            Vh = Vh.to(self.initial_user_dict.device)
            # U, S, Vh = torch.linalg.svd(self.initial_user_dict, full_matrices=False)
            universal_dict = Vh[:self.k_universal, :].detach()  # shape: [k_universal, dict_dim]

        # For each sample, get its own user-specific row.
        user_specific = self.initial_user_dict[user_id]  # [B, dict_dim]

        # Expand the universal dictionary (computed globally) to match the batch size.
        universal_dict_expanded = universal_dict.unsqueeze(0).expand(user_specific.shape[0], -1, -1)  # [B, k_universal, dict_dim]

        # Unsqueeze the personal vector to shape [B, 1, dict_dim].
        user_specific_unsq = user_specific.unsqueeze(1)  # [B, 1, dict_dim]

        # Concatenate to form the effective dictionary: shape [B, k_universal+1, dict_dim].
        effective_dictionary = torch.cat([universal_dict_expanded, user_specific_unsq], dim=1)


        # Compute bias from personality and trial features.
        dict_bias = self.dict_mlp(torch.cat([fused_per, trial_features], dim=1))  # [B, dict_dim]
        effective_dictionary = effective_dictionary + dict_bias.unsqueeze(1)  # [B, k+1, dict_dim]

        # --- User Selection/Gating ---
        # 6) User selection/gating
        fused_tokens_avg = fused_tokens.mean(dim=1)
        selector_logits = self.user_selector(fused_tokens_avg)  # now outputs (k_universal+1) logits
        user_attn_weights = sparsemax(selector_logits, dim=-1)
        user_embedding = torch.matmul(user_attn_weights.unsqueeze(1), effective_dictionary).squeeze(1)  # [B, dict_dim]




        # 7) Adversarial user classifier
        user_logits_before = self.user_classifier_before(fused_per)
        user_logits_after  = self.user_classifier_after(user_embedding)

        # 8) Predict personality
        if self.use_personality:
            if self.use_stim_emo:
                concat_in = torch.cat([user_embedding, stim_emo], dim=1)
            else:
                concat_in = user_embedding
            if isinstance(concat_in, torch.Tensor) and len(concat_in.shape) == 1:
                concat_in = concat_in.unsqueeze(0)
            pred_personality = self.rawpers_fc(concat_in)
        else:
            batch_size = fused_per.size(0)
            pred_personality = torch.zeros(batch_size, self.personality_dim, device=fused_per.device)

        # 9) Optional gender prediction
        gender_pred = None
        if self.use_gender:
            if self.use_stim_emo:
                gender_in = torch.cat([user_embedding, stim_emo, trial_features], dim=1)
            else:
                gender_in = torch.cat([user_embedding, trial_features], dim=1)
            gender_pred = self.gender_branch(gender_in)

        # 10) Prepare for emotion branch (fusion transformer)
        personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
        scale_factor = 2.0
        personality_tokens = scale_factor * personality_tokens
        replication_factor = 3
        personality_tokens = personality_tokens.repeat(1, replication_factor, 1)
        fused_tokens_with_personality = torch.cat([personality_tokens, fused_tokens], dim=1)
        mod_tokens = self.modal_transformer(fused_tokens_with_personality)

        x_emo = self.hierarchical_conv_emotion(mod_tokens.transpose(1, 2))
        x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)
        x_emo_flat = x_emo_pooled.flatten(start_dim=1)
        fused_emo = self.fuse_fc_emo(x_emo_flat)
        personality_from_emotion = self.emotion_to_personality(fused_emo)

        # 11) Emotion Branch with Attention over Signals
        # Prepare tokens for attention: fused_emo_full, trial_features, fused_per, and optionally stim_emo.
        fused_emo_full = torch.cat([user_embedding, fused_emo, rep_emotion], dim=1)
        fused_emo_full = self.emo_per_fc(fused_emo_full)
        tokens = [fused_emo_full, trial_features, fused_per]
        if stim_emo is not None:
            tokens.append(stim_emo)
        # Now, the attention module outputs a weighted sum with shape [B, fused_input_dim]
        final_emotion_repr, attn_weights = self.emotion_attn(tokens)
        
        # 12) Emotion logit heads: feed the attended representation and pred_personality
        emotion_logits_list = []
        for head in self.emotion_heads:
            logits = head(final_emotion_repr, pred_personality)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)
        
        # Gradient Reversal for user-agnostic emotion
        rev_user = self.grad_reverse(fused_emo)
        user_logits = self.user_classifier(rev_user)
        
        # 13) Additional user-related losses and regularization
        user_loss_before = F.cross_entropy(user_logits_before, user_id)
        user_loss_after  = F.cross_entropy(user_logits_after, user_id)
        user_loss_compare = F.cross_entropy(user_logits_after, user_logits_before.argmax(dim=1))
        user_loss_all = (0.1 * user_loss_before) + (0.1 * user_loss_after) + (0.8 * user_loss_compare)
        attn_entropy = -torch.sum(user_attn_weights * torch.log(user_attn_weights + 1e-8), dim=1).mean()
        sparse_loss = self.sparse_attn_coef * attn_entropy
        consistency_loss = 1 - F.cosine_similarity(pred_personality, personality_from_emotion, dim=1).mean()
        combined_loss = sparse_loss + self.lambda_consistency * consistency_loss + user_loss_all * self.user_loss_weight

        return (combined_loss, 
                pred_personality, 
                emotion_logits, 
                fused_per, 
                gender_pred, 
                user_logits)

def scale_with_sigmoid(x, max_val):
    return (torch.sigmoid(x) - 0.5) * 0.1 * max_val


##############################################
# Attention Pooling Module for Trial Pooling
##############################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Positional Encoding (for modality sequences)
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape [1, max_len, d_model]

    def forward(self, x):
        # x shape: [B, T, d_model]
        return x + self.pe[:, :x.size(1), :]

# ------------------------------
# Downsampling helper (deterministic, e.g., simple interpolation)
# ------------------------------
def downsample_sequence_deterministic(x, target_length):
    # x shape: [B, T, d]
    B, T, d = x.shape
    if T == target_length:
        return x
    # Use F.interpolate (requires x to be of shape [B, d, T])
    x = x.transpose(1, 2)  # now [B, d, T]
    x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
    return x.transpose(1, 2)  # back to [B, target_length, d]

# ------------------------------
# Attention Pooling Module for Trials
# ------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: [B, num_trials, input_dim]
        attn_scores = torch.tanh(self.fc1(x))         # [B, num_trials, hidden_dim]
        attn_scores = self.fc2(attn_scores)             # [B, num_trials, 1]
        attn_weights = F.softmax(attn_scores, dim=1)     # [B, num_trials, 1]
        pooled = torch.sum(x * attn_weights, dim=1)      # [B, input_dim]
        return pooled, attn_weights

# ------------------------------
# (Reused) Gradient Reversal for adversarial loss
# ------------------------------
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradReverseLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

# ------------------------------
# (Reused) Emotion Signal Attention module
# ------------------------------
class EmotionSignalAttention(nn.Module):
    def __init__(self, input_dims, common_dim, hidden_dim):
        super().__init__()
        self.num_tokens = len(input_dims)
        self.projections = nn.ModuleList([nn.Linear(dim, common_dim) for dim in input_dims])
        self.attn_fc = nn.Linear(common_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
    
    def forward(self, tokens):
        # tokens: list of tensors, each [B, dim]
        projected_tokens = [proj(token) for proj, token in zip(self.projections, tokens)]
        tokens_stack = torch.stack(projected_tokens, dim=1)  # [B, num_tokens, common_dim]
        hidden = torch.tanh(self.attn_fc(tokens_stack))        # [B, num_tokens, hidden_dim]
        scores = self.attn_score(hidden)                       # [B, num_tokens, 1]
        attn_weights = F.softmax(scores, dim=1)                # [B, num_tokens, 1]
        weighted_sum = torch.sum(tokens_stack * attn_weights, dim=1)  # [B, common_dim]
        return weighted_sum, attn_weights.squeeze(-1)

# ------------------------------
# Advanced Multi–Trial Fusion Model (Updated)
# ------------------------------
class AdvancedMultiTrialFusionModel(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
                 embedding_noise_std=0.0, max_val=1.0,
                 emotion_num_labels=4,
                 use_gender=False,
                 use_stim_emo=True,
                 use_personality=True,
                 sparse_attn_coef=0.01,
                 base_dict_size=10, ext_dict_size=10):
        super().__init__()
        # Basic flags and hyperparameters
        self.use_personality = use_personality
        self.use_stim_emo = use_stim_emo
        self.use_gender = use_gender
        self.target_length = target_length  # fixed length for each trial after downsampling
        self.personality_dim = personality_dim
        self.sparse_attn_coef = sparse_attn_coef
        self.lambda_consistency = 0.5
        self.user_loss_weight = 1
        
        # For multi-trial inputs, number of trials is assumed to be 6.
        self.num_trials = 6
        
        # Define dimensions for small modality branches.
        self.transformer_small_dim = 64
        self.transformer_small_head = 2
        
        # ------------------------------
        # 1) Modality-specific modules (processing per trial)
        # ------------------------------
        self.eye_proj = nn.Linear(eye_dim, self.transformer_small_dim)
        self.pupil_proj = nn.Linear(pupil_dim, self.transformer_small_dim)
        self.au_proj = nn.Linear(au_dim, self.transformer_small_dim)
        self.gsr_proj = nn.Linear(gsr_dim, self.transformer_small_dim)
        
        self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)
        
        def make_small_transformer(d_model, nhead):
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.3, batch_first=True),
                num_layers=1
            )
        self.eye_transformer = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.pupil_transformer = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.au_transformer = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.gsr_transformer = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        
        self.project_modality = nn.Linear(self.transformer_small_dim, proj_dim)
        # Positional embedding for fusion over time per trial
        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, proj_dim * 4))
        
        # ------------------------------
        # 2) Trial-level fusion and attention pooling
        # ------------------------------
        # After processing each trial independently, we concatenate modality projections
        # and obtain a trial-level feature of dimension = proj_dim * 4.
        self.fuse_fc = nn.Linear(proj_dim * 4, proj_dim * 4)
        self.trial_attention = AttentionPooling(input_dim=proj_dim * 4, hidden_dim=64)
        self.features_dim=32
        # ------------------------------
        # 3) Dictionary Block for user adaptation
        # ------------------------------
        self.dict_dim = 256
        self.initial_user_dict = nn.Parameter(torch.randn(num_users, self.dict_dim))
        self.k_universal = 10
        # For dictionary modulation, we assume an auxiliary feature dimension (e.g. 32)
        self.dict_mlp = nn.Sequential(
            nn.Linear(proj_dim * 4 + self.features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.dict_dim)
        )
        self.user_selector = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.k_universal + 1)
        )
        
        # ------------------------------
        # 4) Personality Branch
        # ------------------------------
        base_personality_in_dim = self.dict_dim + (stim_emo_dim if self.use_stim_emo else 0)
        self.rawpers_fc = nn.Sequential(
            nn.Linear(base_personality_in_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_person),
            nn.Linear(proj_dim * 2, personality_dim)
        )
        
        # ------------------------------
        # 5) Fusion for Emotion Prediction
        # ------------------------------
        self.modal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=proj_dim * 4, nhead=transformer_nhead, dropout=0.3, batch_first=True),
            num_layers=transformer_layers
        )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(proj_dim * 4, proj_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(proj_dim * 4),
            nn.ReLU(),
            nn.Conv1d(proj_dim * 4, proj_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(proj_dim * 4),
            nn.ReLU()
        )
        self.fixed_T_emo = 8 if stim_emo_dim == 0 else 10
        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(proj_dim * 4 * self.fixed_T_emo, proj_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )
        self.emotion_to_personality = nn.Linear(proj_dim * 4, personality_dim)
        self.personality_token_proj = nn.Linear(1, proj_dim * 4)
        
        emotion_attn_input_dims = [proj_dim * 4, 32, 128] + ([stim_emo_dim] if self.use_stim_emo else [])
        common_dim = sum(emotion_attn_input_dims)  # This should be 128 + 32 + 128 + 6 = 294.



        self.emotion_attn = EmotionSignalAttention(
                    input_dims=emotion_attn_input_dims,
                    common_dim=common_dim,
                    hidden_dim=64
                )
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead(common_dim, personality_dim, proj_dim * 4, emotion_num_classes, dropout_rate=dropout_emotion)
            for _ in range(emotion_num_labels)
        ])
        

        self.emotion_pool_proj = nn.Linear(proj_dim * 4, 32)  # Project pooled_repr (of size proj_dim*4) to 32 dims
        self.user_embed_proj = nn.Linear(self.dict_dim, 128)   # Project user_embedding (of size dict_dim, e.g. 256) to 128 dims

        # ------------------------------
        # 6) Adversarial User Classifier and Optional Gender Branch
        # ------------------------------
        self.grad_reverse = GradReverseLayer(lambda_=1.0)
        self.user_classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )
        self.user_classifier_before = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )
        self.user_classifier_after = nn.Sequential(
            nn.Linear(self.dict_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )
        if self.use_gender:
            gender_input_dim = base_personality_in_dim   # 256 + 6 = 262
            self.gender_branch = nn.Sequential(
                nn.Linear(gender_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )

        else:
            self.gender_branch = None

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in, 
                eye_feat, au_feat, shimmer_feat):
        """
        Input shapes:
          - eye, pupil, au, gsr: [B, N, T, dim] where N = number of trials (expected 6)
          - stim_emo: [B, N, stim_emo_dim] (if used)
          - The other inputs remain as before.
        """
        B, N, T, _ = eye.shape  # N is expected to be 6
        # Process each modality per trial:
        # Reshape to merge batch and trial dimensions.
        eye = eye.view(B * N, T, -1)
        pupil = pupil.view(B * N, T, -1)
        au = au.view(B * N, T, -1)
        gsr = gsr.view(B * N, T, -1)
        
        # Projection and positional encoding:
        eye_emb = self.pos_encoding_modality(self.eye_proj(eye))
        pupil_emb = self.pos_encoding_modality(self.pupil_proj(pupil))
        au_emb = self.pos_encoding_modality(self.au_proj(au))
        gsr_emb = self.pos_encoding_modality(self.gsr_proj(gsr))
        
        # Pass through small transformer encoders:
        eye_tok = self.eye_transformer(eye_emb)
        pupil_tok = self.pupil_transformer(pupil_emb)
        au_tok = self.au_transformer(au_emb)
        gsr_tok = self.gsr_transformer(gsr_emb)
        
        # Downsample each trial's sequence to fixed length.
        eye_tok = downsample_sequence_deterministic(eye_tok, self.target_length)
        pupil_tok = downsample_sequence_deterministic(pupil_tok, self.target_length)
        au_tok = downsample_sequence_deterministic(au_tok, self.target_length)
        gsr_tok = downsample_sequence_deterministic(gsr_tok, self.target_length)
        
        # Project each modality for fusion.
        eye_proj_out = self.project_modality(eye_tok)
        pupil_proj_out = self.project_modality(pupil_tok)
        au_proj_out = self.project_modality(au_tok)
        gsr_proj_out = self.project_modality(gsr_tok)
        
        # Concatenate along feature dimension.
        raw_mod_tokens = torch.cat([eye_proj_out, pupil_proj_out, au_proj_out, gsr_proj_out], dim=-1)  # [B*N, target_length, proj_dim*4]
        raw_mod_tokens = raw_mod_tokens + self.pos_embedding[:, :raw_mod_tokens.size(1), :]
        # Average over time dimension to obtain a trial-level feature.
        trial_repr = raw_mod_tokens.mean(dim=1)  # [B*N, proj_dim*4]
        # Reshape back to [B, N, proj_dim*4]
        trial_repr = trial_repr.view(B, N, -1)
        # Pool across trials using attention.
        pooled_repr, trial_attn_weights = self.trial_attention(trial_repr)  # [B, proj_dim*4]
        
        # ------------------------------
        # Dictionary Block for User Adaptation
        # ------------------------------
        # Compute a universal dictionary via SVD on initial_user_dict.
                # ------------------------------
        # Dictionary Block for User Adaptation
        # ------------------------------
        with torch.no_grad():
            initial_dict_cpu = self.initial_user_dict.cpu()
            U, S, Vh = torch.linalg.svd(initial_dict_cpu, full_matrices=False)
            Vh = Vh.to(self.initial_user_dict.device)
            universal_dict = Vh[:self.k_universal, :].detach()  # shape: [k_universal, dict_dim] (dict_dim=256)
        universal_dict_expanded = universal_dict.unsqueeze(0).expand(B, -1, -1)  # [B, k_universal, 256]
        
        # Ensure user_id is a single value per sample:
        if user_id.dim() > 1:
            user_id = user_id[:, 0]  # now shape becomes [B]
        user_specific = self.initial_user_dict[user_id]  # [B, 256]
        user_specific_unsq = user_specific.unsqueeze(1)    # [B, 1, 256]
        effective_dictionary = torch.cat([universal_dict_expanded, user_specific_unsq], dim=1)  # [B, k_universal+1, 256]

        selector_logits = self.user_selector(pooled_repr)
        user_attn_weights_dict = F.softmax(selector_logits, dim=-1)
        user_embedding = torch.matmul(user_attn_weights_dict.unsqueeze(1), effective_dictionary).squeeze(1)  # [B, dict_dim]
        
        # ------------------------------
        # Personality Prediction
        # ------------------------------
        if self.use_personality:
            if self.use_stim_emo:
                # Average stim_emo over trials.
                stim_emo_avg = stim_emo.mean(dim=1)  # [B, stim_emo_dim]
                concat_in = torch.cat([user_embedding, stim_emo_avg], dim=1)
            else:
                concat_in = user_embedding
            pred_personality = self.rawpers_fc(concat_in)
        else:
            pred_personality = torch.zeros(B, self.personality_dim, device=pooled_repr.device)
        
        # ------------------------------
        # Emotion Branch
        # ------------------------------
        # Create personality tokens.
        personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
        personality_tokens = 2.0 * personality_tokens
        personality_tokens = personality_tokens.repeat(1, 3, 1)  # replication factor 3
        # For emotion, we use pooled_repr repeated over a (fake) time dimension.
        fused_tokens_with_personality = torch.cat([personality_tokens, pooled_repr.unsqueeze(1).repeat(1, self.target_length, 1)], dim=1)
        mod_tokens = self.modal_transformer(fused_tokens_with_personality)
        x_emo = self.hierarchical_conv_emotion(mod_tokens.transpose(1, 2))
        x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)

        x_emo_flat = x_emo_pooled.flatten(start_dim=1)
        fused_emo = self.fuse_fc_emo(x_emo_flat)
        personality_from_emotion = self.emotion_to_personality(fused_emo)
        # Prepare tokens for emotion attention.
        pooled_repr_proj = self.emotion_pool_proj(pooled_repr)    # [B, 32]
        user_embedding_proj = self.user_embed_proj(user_embedding)  # [B, 128]

        if self.use_stim_emo:
            stim_emo_avg = stim_emo.mean(dim=1)
            tokens = [fused_emo, pooled_repr_proj, user_embedding_proj, stim_emo_avg]
        else:
            tokens = [fused_emo, pooled_repr_proj, user_embedding_proj]
        final_emotion_repr, attn_weights_emotion = self.emotion_attn(tokens)
        emotion_logits_list = []
        for head in self.emotion_heads:
            logits = head(final_emotion_repr, pred_personality)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)
        
        # ------------------------------
        # Adversarial User Classifier and Gender Branch
        # ------------------------------
        rev_user = self.grad_reverse(fused_emo)
        user_logits = self.user_classifier(rev_user)
        user_logits_before = self.user_classifier_before(pooled_repr)
        user_logits_after = self.user_classifier_after(user_embedding)
        # user_loss_before = F.cross_entropy(user_logits_before, user_id)
        user_loss_after  = F.cross_entropy(user_logits_after, user_id)
        # user_loss_compare = F.cross_entropy(user_logits_after, user_logits_before.argmax(dim=1))
        user_loss_all =   user_loss_after
        attn_entropy = -torch.sum(trial_attn_weights * torch.log(trial_attn_weights + 1e-8), dim=1).mean()
        sparse_loss = self.sparse_attn_coef * attn_entropy
        # consistency_loss = 1 - F.cosine_similarity(pred_personality, personality_from_emotion, dim=1).mean()
        combined_loss = sparse_loss + user_loss_all * self.user_loss_weight
        
        # Optional Gender Prediction.
        gender_pred = None
        if self.use_gender:
            if self.use_stim_emo:
                stim_emo_avg = stim_emo.mean(dim=1)
                gender_in = torch.cat([user_embedding, stim_emo_avg], dim=1)
            else:
                gender_in = user_embedding
            gender_pred = self.gender_branch(gender_in)
        
        return (combined_loss, pred_personality, emotion_logits, pooled_repr, gender_pred, user_logits)
    
 # Advanced Multi–Trial Fusion Model (Updated)
# ------------------------------
class AdvancedMultiTrialFusionModelBINNED(nn.Module):
    def __init__(self,
                 eye_dim, pupil_dim, au_dim, gsr_dim,
                 stim_emo_dim, hidden_dim, target_length,
                 personality_dim, emotion_num_classes, num_users,
                 eye_feat_dim, au_feat_dim, shimmer_feat_dim,
                 proj_dim=32, transformer_nhead=4, transformer_layers=1,
                 dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
                 embedding_noise_std=0.0, max_val=1.0,
                 emotion_num_labels=4,
                 use_gender=False,
                 use_stim_emo=True,
                 use_personality=True,
                 sparse_attn_coef=0.01,
                 base_dict_size=10, ext_dict_size=10):
        super().__init__()
        # Basic flags and hyperparameters
        self.use_personality = use_personality
        self.use_stim_emo = use_stim_emo
        self.use_gender = use_gender
        self.target_length = target_length  # fixed length for each trial after downsampling
        self.personality_dim = personality_dim
        self.sparse_attn_coef = sparse_attn_coef
        self.lambda_consistency = 0.5
        self.user_loss_weight = 1
        
        # For multi-trial inputs, number of trials is assumed to be 6.
        self.num_trials = 6
        
        # Define dimensions for small modality branches.
        self.transformer_small_dim = 64
        self.transformer_small_head = 2
        
        # ------------------------------
        # 1) Modality-specific modules (processing per trial)
        # ------------------------------
        self.eye_proj = nn.Linear(eye_dim, self.transformer_small_dim)
        self.pupil_proj = nn.Linear(pupil_dim, self.transformer_small_dim)
        self.au_proj = nn.Linear(au_dim, self.transformer_small_dim)
        self.gsr_proj = nn.Linear(gsr_dim, self.transformer_small_dim)
        
        self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)
        
        def make_small_transformer(d_model, nhead):
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.3, batch_first=True),
                num_layers=1
            )
        self.eye_transformer = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.pupil_transformer = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.au_transformer = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        self.gsr_transformer = make_small_transformer(self.transformer_small_dim, self.transformer_small_head)
        
        self.project_modality = nn.Linear(self.transformer_small_dim, proj_dim)
        # Positional embedding for fusion over time per trial
        self.pos_embedding = nn.Parameter(torch.randn(1, target_length, proj_dim * 4))
        
        # ------------------------------
        # 2) Trial-level fusion and attention pooling
        # ------------------------------
        # After processing each trial independently, we concatenate modality projections
        # and obtain a trial-level feature of dimension = proj_dim * 4.
        self.fuse_fc = nn.Linear(proj_dim * 4, proj_dim * 4)
        self.trial_attention = AttentionPooling(input_dim=proj_dim * 4, hidden_dim=64)
        self.features_dim=32
        # ------------------------------
        # 3) Dictionary Block for user adaptation
        # ------------------------------
        self.dict_dim = 256
        self.initial_user_dict = nn.Parameter(torch.randn(num_users, self.dict_dim))
        self.k_universal = 10
        # For dictionary modulation, we assume an auxiliary feature dimension (e.g. 32)
        self.dict_mlp = nn.Sequential(
            nn.Linear(proj_dim * 4 + self.features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.dict_dim)
        )
        self.user_selector = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.k_universal + 1)
        )
        
        # ------------------------------
        # 4) Personality Branch
        # ------------------------------
        base_personality_in_dim = self.dict_dim + (stim_emo_dim if self.use_stim_emo else 0)
        self.rawpers_fc = nn.Sequential(
                nn.Linear(base_personality_in_dim, proj_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout_person),
                nn.Linear(proj_dim * 2, 15)  # 15 = 3 classes * 5 personality dimensions
            )


        
        # ------------------------------
        # 5) Fusion for Emotion Prediction
        # ------------------------------
        self.modal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=proj_dim * 4, nhead=transformer_nhead, dropout=0.3, batch_first=True),
            num_layers=transformer_layers
        )
        self.hierarchical_conv_emotion = nn.Sequential(
            nn.Conv1d(proj_dim * 4, proj_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(proj_dim * 4),
            nn.ReLU(),
            nn.Conv1d(proj_dim * 4, proj_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(proj_dim * 4),
            nn.ReLU()
        )
        self.fixed_T_emo = 8 if stim_emo_dim == 0 else 10
        self.fuse_fc_emo = nn.Sequential(
            nn.Linear(proj_dim * 4 * self.fixed_T_emo, proj_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_emotion)
        )
        self.emotion_to_personality = nn.Linear(proj_dim * 4, personality_dim)
        self.personality_token_proj = nn.Linear(1, proj_dim * 4)
        
        emotion_attn_input_dims = [proj_dim * 4, 32, 128] + ([stim_emo_dim] if self.use_stim_emo else [])
        common_dim = sum(emotion_attn_input_dims)  # This should be 128 + 32 + 128 + 6 = 294.



        self.emotion_attn = EmotionSignalAttention(
                    input_dims=emotion_attn_input_dims,
                    common_dim=common_dim,
                    hidden_dim=64
                )
        self.emotion_heads = nn.ModuleList([
            AdvancedEmotionHead(common_dim, personality_dim, proj_dim * 4, emotion_num_classes, dropout_rate=dropout_emotion)
            for _ in range(emotion_num_labels)
        ])
        

        self.emotion_pool_proj = nn.Linear(proj_dim * 4, 32)  # Project pooled_repr (of size proj_dim*4) to 32 dims
        self.user_embed_proj = nn.Linear(self.dict_dim, 128)   # Project user_embedding (of size dict_dim, e.g. 256) to 128 dims

        # ------------------------------
        # 6) Adversarial User Classifier and Optional Gender Branch
        # ------------------------------
        # self.grad_reverse = GradReverseLayer(lambda_=1.0)
        self.user_classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )
        self.user_classifier_before = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )
        self.user_classifier_after = nn.Sequential(
            nn.Linear(self.dict_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_users)
        )
        if self.use_gender:
            gender_input_dim = base_personality_in_dim   # 256 + 6 = 262
            self.gender_branch = nn.Sequential(
                nn.Linear(gender_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )

        else:
            self.gender_branch = None

    def forward(self, eye, pupil, au, gsr, stim_emo, user_id, personality_in, 
                eye_feat, au_feat, shimmer_feat):
        """
        Input shapes:
          - eye, pupil, au, gsr: [B, N, T, dim] where N = number of trials (expected 6)
          - stim_emo: [B, N, stim_emo_dim] (if used)
          - The other inputs remain as before.
        """
        B, N, T, _ = eye.shape  # N is expected to be 6
        # Process each modality per trial:
        # Reshape to merge batch and trial dimensions.
        eye = eye.view(B * N, T, -1)
        pupil = pupil.view(B * N, T, -1)
        au = au.view(B * N, T, -1)
        gsr = gsr.view(B * N, T, -1)
        
        # Projection and positional encoding:
        eye_emb = self.pos_encoding_modality(self.eye_proj(eye))
        pupil_emb = self.pos_encoding_modality(self.pupil_proj(pupil))
        au_emb = self.pos_encoding_modality(self.au_proj(au))
        gsr_emb = self.pos_encoding_modality(self.gsr_proj(gsr))
        
        # Pass through small transformer encoders:
        eye_tok = self.eye_transformer(eye_emb)
        pupil_tok = self.pupil_transformer(pupil_emb)
        au_tok = self.au_transformer(au_emb)
        gsr_tok = self.gsr_transformer(gsr_emb)
        
        # Downsample each trial's sequence to fixed length.
        eye_tok = downsample_sequence_deterministic(eye_tok, self.target_length)
        pupil_tok = downsample_sequence_deterministic(pupil_tok, self.target_length)
        au_tok = downsample_sequence_deterministic(au_tok, self.target_length)
        gsr_tok = downsample_sequence_deterministic(gsr_tok, self.target_length)
        
        # Project each modality for fusion.
        eye_proj_out = self.project_modality(eye_tok)
        pupil_proj_out = self.project_modality(pupil_tok)
        au_proj_out = self.project_modality(au_tok)
        gsr_proj_out = self.project_modality(gsr_tok)
        
        # Concatenate along feature dimension.
        raw_mod_tokens = torch.cat([eye_proj_out, pupil_proj_out, au_proj_out, gsr_proj_out], dim=-1)  # [B*N, target_length, proj_dim*4]
        raw_mod_tokens = raw_mod_tokens + self.pos_embedding[:, :raw_mod_tokens.size(1), :]
        # Average over time dimension to obtain a trial-level feature.
        trial_repr = raw_mod_tokens.mean(dim=1)  # [B*N, proj_dim*4]
        # Reshape back to [B, N, proj_dim*4]
        trial_repr = trial_repr.view(B, N, -1)
        # Pool across trials using attention.
        pooled_repr, trial_attn_weights = self.trial_attention(trial_repr)  # [B, proj_dim*4]
        
        # ------------------------------
        # Dictionary Block for User Adaptation
        # ------------------------------
        # Compute a universal dictionary via SVD on initial_user_dict.
                # ------------------------------
        # Dictionary Block for User Adaptation
        # ------------------------------
        with torch.no_grad():
            initial_dict_cpu = self.initial_user_dict.cpu()
            U, S, Vh = torch.linalg.svd(initial_dict_cpu, full_matrices=False)
            Vh = Vh.to(self.initial_user_dict.device)
            universal_dict = Vh[:self.k_universal, :].detach()  # shape: [k_universal, dict_dim] (dict_dim=256)
        universal_dict_expanded = universal_dict.unsqueeze(0).expand(B, -1, -1)  # [B, k_universal, 256]
        
        # Ensure user_id is a single value per sample:
        if user_id.dim() > 1:
            user_id = user_id[:, 0]  # now shape becomes [B]
        user_specific = self.initial_user_dict[user_id]  # [B, 256]
        user_specific_unsq = user_specific.unsqueeze(1)    # [B, 1, 256]
        effective_dictionary = torch.cat([universal_dict_expanded, user_specific_unsq], dim=1)  # [B, k_universal+1, 256]

        selector_logits = self.user_selector(pooled_repr)
        user_attn_weights_dict = F.softmax(selector_logits, dim=-1)
        user_embedding = torch.matmul(user_attn_weights_dict.unsqueeze(1), effective_dictionary).squeeze(1)  # [B, dict_dim]
        
        # ------------------------------
        # Personality Prediction
        # ------------------------------
        if self.use_personality:
            if self.use_stim_emo:
                # Average stim_emo over trials.
                stim_emo_avg = stim_emo.mean(dim=1)  # [B, stim_emo_dim]
                concat_in = torch.cat([user_embedding, stim_emo_avg], dim=1)
            else:
                concat_in = user_embedding
            pred_personality = self.rawpers_fc(concat_in)
        else:
            pred_personality = torch.zeros(B, self.personality_dim, device=pooled_repr.device)
        
        # ------------------------------
        # Emotion Branch
        # ------------------------------
        # Create personality tokens.
        personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
        personality_tokens = 2.0 * personality_tokens
        personality_tokens = personality_tokens.repeat(1, 3, 1)  # replication factor 3
        # For emotion, we use pooled_repr repeated over a (fake) time dimension.
        fused_tokens_with_personality = torch.cat([personality_tokens, pooled_repr.unsqueeze(1).repeat(1, self.target_length, 1)], dim=1)
        mod_tokens = self.modal_transformer(fused_tokens_with_personality)
        x_emo = self.hierarchical_conv_emotion(mod_tokens.transpose(1, 2))
        x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)

        x_emo_flat = x_emo_pooled.flatten(start_dim=1)
        fused_emo = self.fuse_fc_emo(x_emo_flat)
        personality_from_emotion = self.emotion_to_personality(fused_emo)
        # Prepare tokens for emotion attention.
        pooled_repr_proj = self.emotion_pool_proj(pooled_repr)    # [B, 32]
        user_embedding_proj = self.user_embed_proj(user_embedding)  # [B, 128]

        if self.use_stim_emo:
            stim_emo_avg = stim_emo.mean(dim=1)
            tokens = [fused_emo, pooled_repr_proj, user_embedding_proj, stim_emo_avg]
        else:
            tokens = [fused_emo, pooled_repr_proj, user_embedding_proj]
        final_emotion_repr, attn_weights_emotion = self.emotion_attn(tokens)
        emotion_logits_list = []
        for head in self.emotion_heads:
            logits = head(final_emotion_repr, pred_personality)
            emotion_logits_list.append(logits.unsqueeze(1))
        emotion_logits = torch.cat(emotion_logits_list, dim=1)
        
        # ------------------------------
        # Adversarial User Classifier and Gender Branch
        # ------------------------------
        # rev_user = self.grad_reverse(fused_emo)
        user_logits = self.user_classifier(fused_emo)
        user_logits_before = self.user_classifier_before(pooled_repr)
        user_logits_after = self.user_classifier_after(user_embedding)
        # user_loss_before = F.cross_entropy(user_logits_before, user_id)
        user_loss_after  = F.cross_entropy(user_logits_after, user_id)
        # user_loss_compare = F.cross_entropy(user_logits_after, user_logits_before.argmax(dim=1))
        user_loss_all =   user_loss_after
        attn_entropy = -torch.sum(trial_attn_weights * torch.log(trial_attn_weights + 1e-8), dim=1).mean()
        sparse_loss = self.sparse_attn_coef * attn_entropy
        # consistency_loss = 1 - F.cosine_similarity(pred_personality, personality_from_emotion, dim=1).mean()
        combined_loss = sparse_loss + user_loss_all * self.user_loss_weight
        
        # Optional Gender Prediction.
        gender_pred = None
        if self.use_gender:
            if self.use_stim_emo:
                stim_emo_avg = stim_emo.mean(dim=1)
                gender_in = torch.cat([user_embedding, stim_emo_avg], dim=1)
            else:
                gender_in = user_embedding
            gender_pred = self.gender_branch(gender_in)
        
        return (combined_loss, pred_personality, emotion_logits, pooled_repr, gender_pred, user_logits)   

# from torch.autograd import Function

# class GradientReversal(Function):
#     @staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.lambda_ = lambda_
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg() * ctx.lambda_, None

# class GradReverseLayer(nn.Module):
#     """A layer that applies gradient reversal in the backward pass."""
#     def __init__(self, lambda_=1.0):
#         super().__init__()
#         self.lambda_ = lambda_
    
#     def forward(self, x):
#         return GradientReversal.apply(x, self.lambda_)

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
#                              (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
#         self.register_buffer('pe', pe)
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]


# class TaskAttention(nn.Module):
#     def __init__(self, d_model):
#         super(TaskAttention, self).__init__()
#         # Learnable queries for each task.
#         self.personality_query = nn.Parameter(torch.randn(1, d_model))
#         self.emotion_query = nn.Parameter(torch.randn(1, d_model))
    
#     def forward(self, tokens):
#         # tokens: [B, T, d_model]
#         B, T, d = tokens.shape
#         q_person = self.personality_query.expand(B, 1, d)  # [B, 1, d]
#         q_emotion = self.emotion_query.expand(B, 1, d)     # [B, 1, d]

#         attn_person = torch.softmax(
#             torch.bmm(q_person, tokens.transpose(1,2)) / math.sqrt(d),
#             dim=-1
#         )  # [B, 1, T]
#         rep_person = torch.bmm(attn_person, tokens)  # [B, 1, d]

#         attn_emotion = torch.softmax(
#             torch.bmm(q_emotion, tokens.transpose(1,2)) / math.sqrt(d),
#             dim=-1
#         )  # [B, 1, T]
#         rep_emotion = torch.bmm(attn_emotion, tokens)  # [B, 1, d]
#         return rep_person.squeeze(1), rep_emotion.squeeze(1)  # each: [B, d]


# class AdvancedConcatFusionMultiModalModelTransformerAttention_new(nn.Module):
#     def __init__(self,
#                  eye_dim, pupil_dim, au_dim, gsr_dim,
#                  stim_emo_dim, hidden_dim, target_length,
#                  personality_dim, emotion_num_classes, num_users,
#                  eye_feat_dim, au_feat_dim, shimmer_feat_dim,
#                  proj_dim=32, transformer_nhead=4, transformer_layers=1,
#                  dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
#                  embedding_noise_std=0.0, max_val=1.0,
#                  emotion_num_labels=4,
#                  use_gender=False,
#                  use_stim_emo=True,
#                  use_personality=True,
#                  # --- New adversarial parameters ---
#                  adversarial_lambda=1.0,   # scale factor for gradient reversal
#                  user_discriminator_hidden=32
#                  ):
#         super(AdvancedConcatFusionMultiModalModelTransformerAttention_new, self).__init__()
        
#         # Cast to int
#         hidden_dim = int(hidden_dim)
#         proj_dim = int(proj_dim)
#         target_length = int(target_length)
#         personality_dim = int(personality_dim)
#         emotion_num_classes = int(emotion_num_classes)
#         transformer_nhead = int(transformer_nhead)
#         transformer_layers = int(transformer_layers)

#         self.stim_emo_dim = stim_emo_dim
#         self.embedding_noise_std = embedding_noise_std
#         self.personality_dim = personality_dim
#         self.max_val = max_val
#         self.target_length = target_length
#         self.finetune_mode = False
#         self.emotion_num_labels = emotion_num_labels
#         self.use_gender = use_gender
#         self.use_stim_emo = use_stim_emo
#         self.use_personality = use_personality

#         # For the small modality‐specific transformers
#         self.transformer_small_dim = 64  
#         self.transformer_small_head = 2

#         # ------- Modality Input Projections ------
#         self.eye_proj    = nn.Linear(eye_dim, self.transformer_small_dim)
#         self.pupil_proj  = nn.Linear(pupil_dim, self.transformer_small_dim)
#         self.au_proj     = nn.Linear(au_dim, self.transformer_small_dim)
#         self.gsr_proj    = nn.Linear(gsr_dim, self.transformer_small_dim)
#         self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)

#         # ------- Modality-Specific Transformers ------
#         self.eye_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=self.transformer_small_dim,
#                 nhead=self.transformer_small_head,
#                 dropout=0.3,               # increased dropout
#                 batch_first=True
#             ),
#             num_layers=1
#         )
#         self.pupil_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=self.transformer_small_dim,
#                 nhead=self.transformer_small_head,
#                 dropout=0.3,
#                 batch_first=True
#             ),
#             num_layers=1
#         )
#         self.au_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=self.transformer_small_dim,
#                 nhead=self.transformer_small_head,
#                 dropout=0.3,
#                 batch_first=True
#             ),
#             num_layers=1
#         )
#         self.gsr_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=self.transformer_small_dim,
#                 nhead=self.transformer_small_head,
#                 dropout=0.3,
#                 batch_first=True
#             ),
#             num_layers=1
#         )
#         self.project_modality = nn.Linear(self.transformer_small_dim, proj_dim)

#         # ------ Fusion Transformer ------
#         self.d_model = proj_dim * 4
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model,
#             nhead=transformer_nhead,
#             dropout=0.3,               # increased dropout
#             batch_first=True
#         )
#         self.modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
#         self.pos_embedding = nn.Parameter(torch.randn(1, target_length, self.d_model))

#         # ------ Hierarchical Convs ------
#         self.hierarchical_conv_personality = nn.Sequential(
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Dropout(0.3),  # extra dropout
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU()
#         )
#         self.hierarchical_conv_emotion = nn.Sequential(
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Dropout(0.3),  # extra dropout
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Dropout(0.3),  # extra dropout
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU()
#         )

#         # Number of frames after the hierarchical conv
#         if stim_emo_dim == 0:
#             self.fixed_T_emo = 8
#         else:
#             self.fixed_T_emo = 10
        
#         self.fuse_fc_emo = nn.Sequential(
#             nn.Linear(self.d_model * self.fixed_T_emo, self.d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout_emotion)
#         )

#         # Personality dimension is half of self.d_model
#         self.d_personlity = self.d_model // 2
#         self.fuse_fc_per = nn.Sequential(
#             nn.Linear(self.d_model, self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person)
#         )

#         # Task Attention
#         self.task_attention = TaskAttention(d_model=self.d_model)
#         self.personality_attention_proj = nn.Linear(self.d_model, self.d_personlity)

#         # Trial-level features
#         self.features_dim = 32
#         self.eye_feat_mlp = nn.Sequential(
#             nn.Linear(eye_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.au_feat_mlp = nn.Sequential(
#             nn.Linear(au_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.shimmer_feat_mlp = nn.Sequential(
#             nn.Linear(shimmer_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.features_mlps = nn.Sequential(
#             nn.Linear(self.features_dim * 3, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.fuse_personality = nn.Linear(2 * self.d_model, self.d_model)
#         self.personality_token_proj = nn.Linear(1, self.d_model)

#         # Personality heads
#         self.emo_per_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + 2 * self.d_model, self.d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout_emotion),
#             nn.Linear(self.d_model, self.d_model)
#         )
#         self.trial_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, personality_dim)
#         )
#         self.rawpers_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0), self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, personality_dim)
#         )
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(self.d_personlity + personality_dim, self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person)
#         )

#         # Optional Gender Branch
#         if self.use_gender:
#             gender_input_dim = self.d_personlity + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0)
#             gender_hidden_dim = 32  
#             self.gender_branch = nn.Sequential(
#                 nn.Linear(gender_input_dim, gender_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(gender_hidden_dim, 2)
#             )
#         else:
#             self.gender_branch = None

#         # Emotion heads
#         self.emotion_heads = nn.ModuleList([
#             AdvancedEmotionHead(
#                 self.d_model + self.features_dim + (self.stim_emo_dim if self.use_stim_emo else 0),
#                 personality_dim,
#                 self.d_model,
#                 emotion_num_classes,
#                 dropout_rate=dropout_emotion
#             )
#             for _ in range(self.emotion_num_labels)
#         ])

#         # ------------------------------------------------------
#         #   (A) Adversarial User Discriminator
#         # ------------------------------------------------------
#         self.grad_reverse = GradReverseLayer(lambda_=adversarial_lambda)
#         self.user_classifier = nn.Sequential(
#             nn.Linear(self.d_personlity, user_discriminator_hidden),
#             nn.ReLU(),
#             nn.Dropout(p=0.4),  # you can tune dropout
#             nn.Linear(user_discriminator_hidden, num_users)
#         )
#         # This will classify user_id from the personality embedding (or any other mid-layer embedding).

#     def forward(self,
#                 eye, pupil, au, gsr,
#                 stim_emo, user_id, personality_in,
#                 eye_feat, au_feat, shimmer_feat):
#         # If we do not use stim_emo, ignore it
#         if not self.use_stim_emo:
#             stim_emo = None

#         # --- (1) Process the four modalities ---
#         eye_emb = self.eye_proj(eye)
#         eye_emb = self.pos_encoding_modality(eye_emb)
#         eye_out = self.eye_transformer(eye_emb)
#         eye_tok = self.project_modality(downsample_sequence_deterministic(eye_out, self.target_length))

#         pupil_emb = self.pupil_proj(pupil)
#         pupil_emb = self.pos_encoding_modality(pupil_emb)
#         pupil_out = self.pupil_transformer(pupil_emb)
#         pupil_tok = self.project_modality(downsample_sequence_deterministic(pupil_out, self.target_length))

#         au_emb = self.au_proj(au)
#         au_emb = self.pos_encoding_modality(au_emb)
#         au_out = self.au_transformer(au_emb)
#         au_tok = self.project_modality(downsample_sequence_deterministic(au_out, self.target_length))

#         gsr_emb = self.gsr_proj(gsr)
#         gsr_emb = self.pos_encoding_modality(gsr_emb)
#         gsr_out = self.gsr_transformer(gsr_emb)
#         gsr_tok = self.project_modality(downsample_sequence_deterministic(gsr_out, self.target_length))
        
#         # Concatenate the per-modality tokens
#         raw_mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)  # [B, T, d_model]
#         raw_mod_tokens = raw_mod_tokens + self.pos_embedding

#         # --- (2) Personality Branch ---
#         # Hierarchical conv
#         x = raw_mod_tokens.transpose(1, 2)  # [B, d_model, T]
#         x_per = self.hierarchical_conv_personality(x)  # [B, d_model, T_per]
#         fused_per = self.fuse_fc_per(x_per.mean(dim=2))  # [B, d_personlity]

#         # Upsample & fuse
#         x_per_upsampled = F.interpolate(x_per, size=raw_mod_tokens.size(1), mode='linear', align_corners=False)
#         x_per_upsampled = x_per_upsampled.transpose(1, 2)  # [B, T, d_model]
#         if not self.use_personality:
#             x_per_upsampled = torch.zeros_like(x_per_upsampled)

#         fused_tokens = torch.cat([raw_mod_tokens, x_per_upsampled], dim=-1)  # [B, T, 2*d_model]
#         fused_tokens = self.fuse_personality(fused_tokens)                  # [B, T, d_model]

#         # Task Attention
#         rep_person, rep_emotion = self.task_attention(fused_tokens)  # [B, d_model] each
#         rep_person_proj = self.personality_attention_proj(rep_person) # [B, d_personlity]
#         final_person_rep = fused_per + rep_person_proj                # [B, d_personlity]

#         # --- (3) Trial-Level Features & Personality Prediction ---
#         eye_trial = self.eye_feat_mlp(eye_feat)
#         au_trial = self.au_feat_mlp(au_feat)
#         shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
#         trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
#         trial_features = self.features_mlps(trial_concat)

#         if self.use_personality:
#             if self.use_stim_emo:
#                 concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
#             else:
#                 concat_in = torch.cat([fused_per, trial_features], dim=1)

#             pred_personality = self.rawpers_fc(concat_in)
#             if self.finetune_mode:
#                 trial_delta = self.trial_fc(concat_in)
#                 trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
#             else:
#                 trial_delta = torch.zeros_like(pred_personality)
#             golden_personality = pred_personality + trial_delta
#         else:
#             batch_size = fused_per.size(0)
#             pred_personality = torch.zeros(batch_size, self.personality_dim, device=fused_per.device)
#             trial_delta = torch.zeros_like(pred_personality)
#             golden_personality = pred_personality
        
#         # Optional Gender Prediction
#         gender_pred = None
#         if self.use_gender:
#             gender_pred = self.gender_branch(concat_in)

#         # --- (4) Personality Tokens for the Fusion Transformer ---
#         personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))  # [B, d_personality, 1] -> [B, d_model, 1] 
#         scale_factor = 2.0
#         personality_tokens = scale_factor * personality_tokens
#         replication_factor = 3
#         personality_tokens = personality_tokens.repeat(1, replication_factor, 1)  # e.g. [B, 3, d_model]
#         fused_tokens_with_personality = torch.cat([personality_tokens, fused_tokens], dim=1)  # [B, 3 + T, d_model]
#         mod_tokens = self.modal_transformer(fused_tokens_with_personality)

#         # --- (5) Emotion Branch ---
#         x_emo = self.hierarchical_conv_emotion(mod_tokens.transpose(1, 2))
#         # Handle pooling
#         x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)
#         x_emo_flat = x_emo_pooled.flatten(start_dim=1)
#         fused_emo = self.fuse_fc_emo(x_emo_flat)
#         fused_emo = self.emo_per_fc(torch.cat([final_person_rep, fused_emo, rep_emotion], dim=1))

#         if stim_emo is not None:
#             concat_in_emo = torch.cat([fused_emo, trial_features, stim_emo], dim=1)
#         else:
#             concat_in_emo = torch.cat([fused_emo, trial_features], dim=1)

#         emotion_logits_list = []
#         for head in self.emotion_heads:
#             logits = head(concat_in_emo, golden_personality)
#             emotion_logits_list.append(logits.unsqueeze(1))
#         emotion_logits = torch.cat(emotion_logits_list, dim=1)  # [B, emotion_num_labels, ...]

#         # ------------------------------------------------------
#         #   (B) Adversarial User Classification
#         # ------------------------------------------------------
#         # Let's use final_person_rep (or fused_per) as the user "embedding."
#         # Pass it through gradient reversal, then classify user_id.
#         reversed_rep = self.grad_reverse(final_person_rep)      # gradient gets reversed
#         user_logits = self.user_classifier(reversed_rep)        # [B, num_users]

#         # Return user_logits so you can compute adversarial loss
#         return (trial_delta,
#                 pred_personality,
#                 emotion_logits,
#                 fused_per,      # or final_person_rep
#                 gender_pred,
#                 user_logits     # add this for adversarial branch
#                 )
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from torch.autograd import Function

# ##################################
# # 1. Gradient Reversal Utilities
# ##################################
# class GradientReversal(Function):
#     @staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.lambda_ = lambda_
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Multiply gradient by -lambda in backward
#         return grad_output.neg() * ctx.lambda_, None

# class GradReverseLayer(nn.Module):
#     """A layer that applies gradient reversal in the backward pass."""
#     def __init__(self, lambda_=1.0):
#         super().__init__()
#         self.lambda_ = lambda_
    
#     def forward(self, x):
#         return GradientReversal.apply(x, self.lambda_)

# ########################################
# # PositionalEncoding, TaskAttention...
# ########################################
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
#                              (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:, : x.size(1)]

# class TaskAttention(nn.Module):
#     def __init__(self, d_model):
#         super(TaskAttention, self).__init__()
#         # Learnable queries
#         self.personality_query = nn.Parameter(torch.randn(1, d_model))
#         self.emotion_query     = nn.Parameter(torch.randn(1, d_model))
    
#     def forward(self, tokens):
#         # tokens: [B, T, d_model]
#         B, T, d = tokens.shape
#         q_person = self.personality_query.expand(B, 1, d)
#         q_emotion= self.emotion_query.expand(B, 1, d)

#         attn_person = torch.softmax(
#             torch.bmm(q_person, tokens.transpose(1, 2)) / math.sqrt(d), dim=-1
#         )  # [B,1,T]
#         rep_person = torch.bmm(attn_person, tokens)  # [B,1,d]

#         attn_emotion = torch.softmax(
#             torch.bmm(q_emotion, tokens.transpose(1, 2)) / math.sqrt(d), dim=-1
#         )
#         rep_emotion = torch.bmm(attn_emotion, tokens)  # [B,1,d]
#         return rep_person.squeeze(1), rep_emotion.squeeze(1)

# ########################################
# # Your main model with adversarial branch
# ########################################

# class AdvancedConcatFusionMultiModalModelTransformerAttention(nn.Module):
#     def __init__(
#         self,
#         eye_dim, pupil_dim, au_dim, gsr_dim,
#         stim_emo_dim, hidden_dim, target_length,
#         personality_dim, emotion_num_classes, num_users,
#         eye_feat_dim, au_feat_dim, shimmer_feat_dim,
#         proj_dim=32, transformer_nhead=4, transformer_layers=1,
#         dropout_person=0.2, dropout_emotion=0.25, dropout_features=0.2,
#         embedding_noise_std=0.0, max_val=1.0,
#         emotion_num_labels=4,
#         use_gender=False,
#         use_stim_emo=True,
#         use_personality=True,
#         # Adversarial parameters:
#         adversarial_lambda=1.0,  # how strongly to apply gradient reversal
#         user_discriminator_hidden=32,
#     ):
#         super().__init__()
#         # Store flags
#         self.use_gender = use_gender
#         self.use_stim_emo = use_stim_emo
#         self.use_personality = use_personality
#         self.num_users = num_users

#         self.transformer_small_dim = 64
#         self.transformer_small_head= 2
#         self.d_model = proj_dim * 4
#         self.personality_dim = personality_dim
#         self.max_val = max_val
#         self.finetune_mode = False
#         self.emotion_num_labels = emotion_num_labels

#         # 1) Modality projections
#         self.eye_proj    = nn.Linear(eye_dim, self.transformer_small_dim)
#         self.pupil_proj  = nn.Linear(pupil_dim, self.transformer_small_dim)
#         self.au_proj     = nn.Linear(au_dim, self.transformer_small_dim)
#         self.gsr_proj    = nn.Linear(gsr_dim, self.transformer_small_dim)
#         self.pos_encoding_modality = PositionalEncoding(self.transformer_small_dim)

#         # 2) Modality-specific transformers
#         enc_layer_small = nn.TransformerEncoderLayer(
#             d_model=self.transformer_small_dim, nhead=self.transformer_small_head,
#             dropout=0.3, batch_first=True
#         )
#         self.eye_transformer    = nn.TransformerEncoder(enc_layer_small,  num_layers=1)
#         self.pupil_transformer  = nn.TransformerEncoder(enc_layer_small,  num_layers=1)
#         self.au_transformer     = nn.TransformerEncoder(enc_layer_small,  num_layers=1)
#         self.gsr_transformer    = nn.TransformerEncoder(enc_layer_small,  num_layers=1)

#         self.project_modality   = nn.Linear(self.transformer_small_dim, proj_dim)
#         self.pos_embedding      = nn.Parameter(torch.randn(1, target_length, self.d_model))

#         # 3) Fusion transformer
#         enc_layer_fusion = nn.TransformerEncoderLayer(
#             d_model=self.d_model, nhead=transformer_nhead, dropout=0.3, batch_first=True
#         )
#         self.modal_transformer = nn.TransformerEncoder(enc_layer_fusion, num_layers=transformer_layers)

#         # 4) Hierarchical Convs
#         self.hierarchical_conv_personality = nn.Sequential(
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Dropout(0.3),  # extra dropout
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU()
#         )
#         self.hierarchical_conv_emotion = nn.Sequential(
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(self.d_model),
#             nn.ReLU()
#         )

#         if stim_emo_dim == 0:
#             self.fixed_T_emo = 8
#         else:
#             self.fixed_T_emo = 10

#         self.fuse_fc_emo = nn.Sequential(
#             nn.Linear(self.d_model * self.fixed_T_emo, self.d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout_emotion),
#         )

#         self.d_personlity = self.d_model // 2
#         self.fuse_fc_per = nn.Sequential(
#             nn.Linear(self.d_model, self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person)
#         )

#         # Task Attention
#         self.task_attention = TaskAttention(self.d_model)
#         self.personality_attention_proj = nn.Linear(self.d_model, self.d_personlity)

#         # Trial-level features
#         self.features_dim = 32
#         self.eye_feat_mlp = nn.Sequential(
#             nn.Linear(eye_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features),
#         )
#         self.au_feat_mlp = nn.Sequential(
#             nn.Linear(au_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.shimmer_feat_mlp = nn.Sequential(
#             nn.Linear(shimmer_feat_dim, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.features_mlps = nn.Sequential(
#             nn.Linear(self.features_dim * 3, self.features_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_features)
#         )
#         self.fuse_personality = nn.Linear(2 * self.d_model, self.d_model)
#         self.personality_token_proj = nn.Linear(1, self.d_model)

#         # Personality heads
#         self.emo_per_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + 2*self.d_model, self.d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout_emotion),
#             nn.Linear(self.d_model, self.d_model)
#         )
#         self.trial_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + self.features_dim + (stim_emo_dim if use_stim_emo else 0), self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, personality_dim)
#         )
#         self.rawpers_fc = nn.Sequential(
#             nn.Linear(self.d_personlity + self.features_dim + (stim_emo_dim if use_stim_emo else 0), self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, self.d_personlity),
#             nn.ReLU(),
#             nn.Dropout(dropout_person),
#             nn.Linear(self.d_personlity, personality_dim)
#         )

#         # Gender branch (optional)
#         if use_gender:
#             gender_input_dim = self.d_personlity + self.features_dim + (stim_emo_dim if use_stim_emo else 0)
#             gender_hidden_dim= 32
#             self.gender_branch = nn.Sequential(
#                 nn.Linear(gender_input_dim, gender_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(gender_hidden_dim, 2)
#             )
#         else:
#             self.gender_branch = None

#         # (Emotion heads) - your code presumably defines 'AdvancedEmotionHead'
#         # Just be sure to handle them normally
#         self.emotion_heads = nn.ModuleList([
#             AdvancedEmotionHead(
#                 self.d_model + self.features_dim + (stim_emo_dim if use_stim_emo else 0),
#                 personality_dim,
#                 self.d_model,
#                 emotion_num_classes,
#                 dropout_rate=dropout_emotion
#             )
#             for _ in range(emotion_num_labels)
#         ])

#         # -----------------------------
#         # (A) Adversarial User Classifier
#         # -----------------------------
#         self.grad_reverse = GradReverseLayer(lambda_=adversarial_lambda)
#         self.user_classifier = nn.Sequential(
#             nn.Linear(self.d_personlity, user_discriminator_hidden),
#             nn.ReLU(),
#             nn.Dropout(p=0.4),
#             nn.Linear(user_discriminator_hidden, num_users)
#         )

#     def forward(
#         self,
#         eye, pupil, au, gsr,
#         stim_emo, user_id, personality_in,
#         eye_feat, au_feat, shimmer_feat,
#     ):
#         # Possibly ignore stim_emo
#         if not self.use_stim_emo:
#             stim_emo = None

#         # (1) Process Modality
#         eye_emb = self.eye_proj(eye)
#         eye_emb = self.pos_encoding_modality(eye_emb)
#         eye_out = self.eye_transformer(eye_emb)
#         eye_tok = self.project_modality(downsample_sequence_deterministic(eye_out,  self.grad_reverse))

#         pupil_emb = self.pupil_proj(pupil)
#         pupil_emb = self.pos_encoding_modality(pupil_emb)
#         pupil_out = self.pupil_transformer(pupil_emb)
#         pupil_tok= self.project_modality(downsample_sequence_deterministic(pupil_out, self.grad_reverse))

#         au_emb = self.au_proj(au)
#         au_emb = self.pos_encoding_modality(au_emb)
#         au_out = self.au_transformer(au_emb)
#         au_tok = self.project_modality(downsample_sequence_deterministic(au_out, self.grad_reverse))

#         gsr_emb= self.gsr_proj(gsr)
#         gsr_emb= self.pos_encoding_modality(gsr_emb)
#         gsr_out= self.gsr_transformer(gsr_emb)
#         gsr_tok= self.project_modality(downsample_sequence_deterministic(gsr_out, self.grad_reverse))

#         raw_mod_tokens = torch.cat([eye_tok, pupil_tok, au_tok, gsr_tok], dim=-1)  # [B,T,d_model]
#         raw_mod_tokens = raw_mod_tokens + self.pos_embedding  # add positional embedding

#         # (2) Personality branch
#         x = raw_mod_tokens.transpose(1, 2)   # [B, d_model, T]
#         x_per = self.hierarchical_conv_personality(x)  # [B, d_model, T_per]
#         fused_per = self.fuse_fc_per(x_per.mean(dim=2))# [B, d_personlity]

#         x_per_upsampled = F.interpolate(x_per, size=raw_mod_tokens.size(1), mode='linear', align_corners=False)
#         x_per_upsampled = x_per_upsampled.transpose(1, 2) # [B,T,d_model]
#         if not self.use_personality:
#             x_per_upsampled = torch.zeros_like(x_per_upsampled)

#         fused_tokens = torch.cat([raw_mod_tokens, x_per_upsampled], dim=-1)  # [B,T,2*d_model]
#         fused_tokens = self.fuse_personality(fused_tokens)                   # [B,T,d_model]

#         rep_person, rep_emotion = self.task_attention(fused_tokens) # each [B,d_model]
#         rep_person_proj = self.personality_attention_proj(rep_person) # [B,d_personlity]
#         final_person_rep = fused_per + rep_person_proj                # [B,d_personlity]

#         # (3) Trial-level & Personality
#         eye_trial = self.eye_feat_mlp(eye_feat)
#         au_trial  = self.au_feat_mlp(au_feat)
#         shimmer_trial = self.shimmer_feat_mlp(shimmer_feat)
#         trial_concat = torch.cat([eye_trial, au_trial, shimmer_trial], dim=1)
#         trial_features = self.features_mlps(trial_concat)

#         if self.use_personality:
#             if stim_emo is not None:
#                 concat_in = torch.cat([fused_per, trial_features, stim_emo], dim=1)
#             else:
#                 concat_in = torch.cat([fused_per, trial_features], dim=1)
#             pred_personality = self.rawpers_fc(concat_in)
#             if self.finetune_mode:
#                 # optional: further delta
#                 trial_delta = self.trial_fc(concat_in)
#                 trial_delta = scale_with_sigmoid(trial_delta, self.max_val)
#             else:
#                 trial_delta = torch.zeros_like(pred_personality)
#             golden_personality = pred_personality + trial_delta
#         else:
#             batch_size = fused_per.size(0)
#             pred_personality = torch.zeros(batch_size, self.personality_dim, device=fused_per.device)
#             trial_delta = torch.zeros_like(pred_personality)
#             golden_personality = pred_personality

#         # (Optional) Gender
#         gender_pred = None
#         if self.use_gender:
#             gender_pred = self.gender_branch(concat_in)

#         # (4) Personality tokens => fused
#         personality_tokens = self.personality_token_proj(pred_personality.unsqueeze(-1))
#         personality_tokens = 2.0 * personality_tokens
#         personality_tokens = personality_tokens.repeat(1, 3, 1)  # replicate
#         fused_tokens_with_personality = torch.cat([personality_tokens, fused_tokens], dim=1)
#         mod_tokens = self.modal_transformer(fused_tokens_with_personality)

#         # (5) Emotion
#         x_emo = self.hierarchical_conv_emotion(mod_tokens.transpose(1, 2))
#         x_emo_pooled = F.adaptive_avg_pool1d(x_emo.cpu(), output_size=self.fixed_T_emo).to(x_emo.device)
#         x_emo_flat = x_emo_pooled.flatten(start_dim=1)
#         fused_emo = self.fuse_fc_emo(x_emo_flat)
#         fused_emo = self.emo_per_fc(torch.cat([final_person_rep, fused_emo, rep_emotion], dim=1))

#         if stim_emo is not None:
#             concat_in_emo = torch.cat([fused_emo, trial_features, stim_emo], dim=1)
#         else:
#             concat_in_emo = torch.cat([fused_emo, trial_features], dim=1)

#         emotion_logits_list = []
#         for head in self.emotion_heads:
#             logits = head(concat_in_emo, golden_personality)
#             emotion_logits_list.append(logits.unsqueeze(1))
#         emotion_logits = torch.cat(emotion_logits_list, dim=1)  # [B, #emo_labels, #classes]

#         # (A) Adversarial user classification
#         # push final_person_rep (or fused_per) through gradient reversal
#         reversed_rep = self.grad_reverse(final_person_rep)  # shape [B, d_personlity]
#         user_logits = self.user_classifier(reversed_rep)    # [B, num_users]

#         return (
#             trial_delta,
#             pred_personality,
#             emotion_logits,
#             fused_per,
#             gender_pred,
#             user_logits  # new
#         )

# ########################################
# # Utility "downsample_sequence_deterministic" etc.
# ########################################

# def downsample_sequence_deterministic(x, target_len):
#     """
#     Your existing logic for downsample, e.g. 1D interpolation or slice.
#     Stub function for example.
#     x shape: [B,T,d]. 
#     Return shape: [B,target_len,d].
#     """
#     B, T, D = x.shape
#     if T == target_len:
#         return x
#     # naive slice or interpolation:
#     indices = torch.linspace(0, T-1, steps=target_len).long()
#     x_down = torch.stack([x[i, indices, :] for i in range(B)], dim=0)
#     return x_down

# ########################################
# # Suppose "AdvancedEmotionHead" is also defined...
# ########################################

def scale_with_sigmoid(x, max_val):
    return max_val * torch.sigmoid(x)

# ------------------------------
# Simple TCN for modality fusion (as defined earlier)
# ------------------------------
class ModalityTCN(nn.Module):
    """
    A simple Temporal Convolutional Network (TCN) for fusing raw modality signals.
    
    Args:
        input_dim (int): Total input feature dimension (sum of eye_dim, pupil_dim, au_dim, gsr_dim).
        output_dim (int): Output (fused) feature dimension.
        num_layers (int): Number of convolutional layers.
        kernel_size (int): Kernel size for convolutions.
        dropout (float): Dropout rate.
    """
    def __init__(self, input_dim, output_dim, num_layers=2, kernel_size=3, dropout=0.3):
        super(ModalityTCN, self).__init__()
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation  # padding to preserve sequence length
            layers.append(nn.Conv1d(in_channels, output_dim, kernel_size, padding=pad, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = output_dim
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, T, input_dim] --> [B, input_dim, T]
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # back to [B, T, output_dim]
        x = x.transpose(1, 2)
        return x

# ------------------------------
# Updated Simple Personality Model
# ------------------------------
class SimplePersonalityModel(nn.Module):
    def __init__(self, eye_dim, pupil_dim, au_dim, gsr_dim, d_model, personality_dim,
                 num_layers=2, kernel_size=3, dropout=0.3):
        """
        Args:
            eye_dim, pupil_dim, au_dim, gsr_dim: Feature dimensions for each modality.
            d_model: Output dimension of the TCN fusion layer.
            personality_dim: Number of personality traits (output dimension).
            num_layers: Number of TCN layers.
            kernel_size: Kernel size for TCN.
            dropout: Dropout rate for TCN and MLP.
        """
        super(SimplePersonalityModel, self).__init__()
        # Total dimension is the sum of modality feature dimensions.
        self.total_modality_dim = eye_dim + pupil_dim + au_dim + gsr_dim
        self.d_model = d_model
        self.personality_dim = personality_dim
        
        # TCN to fuse modalities.
        self.tcn = ModalityTCN(input_dim=self.total_modality_dim,
                               output_dim=self.d_model,
                               num_layers=num_layers,
                               kernel_size=kernel_size,
                               dropout=dropout)
        # MLP for personality prediction.
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, personality_dim)
        )
    
    def forward(self, eye, pupil, au, gsr):
        """
        Args:
            eye, pupil, au, gsr: Tensors of shape [B, T, feature_dim] for each modality.
        Returns:
            pred_personality: Tensor of shape [B, personality_dim].
        """
        # Concatenate modalities along the feature dimension.
        x = torch.cat([eye, pupil, au, gsr], dim=-1)  # [B, T, total_modality_dim]
        fused_tokens = self.tcn(x)  # [B, T, d_model]
        # Pool over the time dimension.
        x_pooled = fused_tokens.mean(dim=1)  # [B, d_model]
        pred_personality = self.fc(x_pooled)  # [B, personality_dim]
        return pred_personality
