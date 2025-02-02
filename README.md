# HeLoRA

This is the implementation of the paper "HeLoRA: LoRA-heterogeneous Federated Fine-tuning for Foundation Models".

# Abstract
Foundation models (FMs) have achieved state-of-the-art performance across various domains, benefiting from their vast number of parameters and the extensive amount of publicly available training data. However, real-world deployments reveal challenges such as system heterogeneity, where not all devices can handle the complexity of FMs, and emerging privacy concerns that limit the availability of public data. To address these challenges, we propose HeLoRA, a novel approach combining low-rank adaptation (LoRA) with federated learning to enable heterogeneous federated fine-tuning. 
HeLoRA allows clients to fine-tune models with different complexities by adjusting the rank values of LoRA matrices, tailoring the process to each device's capabilities. To tackle the challenge of aggregating models with different structures, HeLoRA introduces two variants, i.e., HeLoRA-Pad and HeLoRA-KD. HeLoRA-Pad employs context-based padding to standardize the LoRA matrices, aligning them with the global model through a rank-based adaptive aggregation strategy. In contrast, HeLoRA-KD leverages the idea of deep mutual learning for aggregation, allowing heterogeneous models to retain their original structures. Extensive experiments with various datasets and ablation studies demonstrate that HeLoRA outperforms existing baselines, promising to enhance the practical deployment of FMs in diverse real-world environments.



