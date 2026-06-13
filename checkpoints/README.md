# Model Checkpoints

This directory contains pre-trained model checkpoints for the CERN-AI unsupervised anomaly detection pipeline.

## jetclass_edgeconv_best.pt
- **Dataset**: JetClass (6M subset)
- **Architecture**: EdgeConv Graph Autoencoder (Dynamic Graph CNN)
- **Parameters**: 37,296
- **Performance**: 0.6808 AUROC (Best result / Flagship model)

## jetclass_gcn_best.pt
- **Dataset**: JetClass (6M subset)
- **Architecture**: GCN Graph Autoencoder (Fixed $k$-NN baseline)
- **Performance**: 0.6541 AUROC (With batched self-loop fix)

## cms_gcn_best.pt
- **Dataset**: CMS Open Data (NanoAOD Run 2)
- **Architecture**: GCN Graph Autoencoder
- **Purpose**: Validation of the graph construction and inference engine on authentic CERN detector data.
