```mermaid
flowchart TD
  %% =========================
  %% Top-level model architecture (CenterSingleObjNet)
  %% =========================

  A["Input image x\n(B, 3, 320, 320)"] --> B["Backbone CNN\nResNet18 feature extractor\nfeature stride = 32"]
  B --> C["Feature map F\n(B, D, 10, 10)"]

  %% -------- Three parallel heads --------
  C --> H1["Center head (1x1 conv)\n(B, 1, 10, 10)\nsigmoid -> center_pred"]
  C --> H2["Box head (1x1 conv)\n(B, 4, 10, 10)\nsigmoid -> box_pred\n[xc,yc,w,h] norm"]
  C --> H3["Class head (1x1 conv)\n(B, 11, 10, 10)\nlogits -> cls_pred"]

  %% -------- Coupling via decode (single-cell selection) --------
  H1 --> D["Decode: pick grid cell\n(i*,j*) = argmax(center_pred[b,0,:,:])"]
  D --> E1["Read box at (i*,j*)\nbox_hat_norm = box_pred[b,:,i*,j*]"]
  D --> E2["Read class at (i*,j*)\ncls_hat = argmax(cls_pred[b,:,i*,j*])"]

  %% -------- Final outputs --------
  E1 --> O["Outputs per image:\ncenter_score, ij_hat, box_hat_norm, cls_hat"]
  E2 --> O



