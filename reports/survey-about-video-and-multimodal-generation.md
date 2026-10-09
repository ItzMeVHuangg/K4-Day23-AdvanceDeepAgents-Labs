# Comprehensive Survey on Video and Multimodal Generation: Architectures, Progress, and Evaluation

## TL;DR
- Video generation has transitioned from traditional heuristic and convolutional models to scalable Diffusion Transformers (DiTs) and advanced visual autoregressive paradigms capable of high-resolution, long-horizon synthesis [1][2][3].
- Modern efficiency breakthroughs leverage spatio-temporal compression via Video-VAEs and adaptive tokenizers alongside attention routing mechanisms like heterogeneous attention and parameterized stripe attention [4][5][6].
- Leading open-source and commercial milestones—such as Open-Sora, CogVideoX, Sora, and Movie Gen—demonstrate unified multimodal integration supporting text-to-video, audio synchronization, and controllable editing [7][8][9][2].
- Robust evaluation requires multidimensional benchmarks (e.g., VBench, AIGVE-60K) and post-training alignment strategies to mitigate temporal drift, low-dynamic bias, and physical grounding errors [10][11][12].

## Background
Video and multimodal generation represents a cornerstone frontier in generative artificial intelligence, uniting spatial visual synthesis with temporal dynamics, audio-visual intelligence, and cross-modal reasoning [1][13]. Historically, video synthesis relied on conditional adversarial networks and recurrence models constrained by low resolution, short temporal horizons, and limited text alignment [3]. The advent of diffusion models and autoregressive transformers revolutionized the field by enabling stable, high-dimensional probability modeling over internet-scale visual data [1][2][3]. 

Foundational work in video diffusion models established how 3D U-Nets and subsequently Diffusion Transformers (DiTs) can model joint spatio-temporal distributions [1][2]. Concurrently, visual autoregressive frameworks and audio-visual intelligence models expanded generation into multi-sensory domains where text, image, video, and audio signals are jointly aligned and generated [14][13]. Understanding these foundations is essential for examining how recent architectural innovations handle the extreme computational and perceptual demands of dynamic visual simulation [15].

## Architectural Paradigms: Diffusion vs. Autoregressive Models
The architectural landscape of video generation is defined primarily by Diffusion Transformers (DiTs) and Autoregressive (AR) models [1][15]. Diffusion models iteratively denoise random Gaussian noise into structured spatio-temporal tensors conditioned on textual or visual prompts [1][2]. While Sora established the viability of DiT backbones acting as world simulators that supersede conventional 3D U-Nets [2], diffusion models face high inference latency due to multi-step iterative denoising [15].

In contrast, autoregressive video generation models predict visual tokens sequentially across spatial and temporal dimensions [14][16]. Recent breakthroughs have sought to overcome the limitations of vector-quantized tokenization: NOVA bypasses vector quantization entirely for superior efficiency and parameter economy [14], while VideoAR implements multi-scale next-scale prediction to ensure robust temporal consistency and scaling behavior [16]. Furthermore, hybrid paradigms such as ARLON couple autoregressive models with diffusion transformers to provide coarse spatial and long-range temporal guidance for high-fidelity long video generation [17].

Cross-modal alignment is equally critical for fusing conditioning signals. Techniques such as AlignMamba utilize optimal transport to enforce local token-level correspondences and global distribution consistency across modalities [18], while contrastive alignment re-weights text tokens based on direct visual correlation [19].

## Spatio-Temporal Efficiency and Latent Compression
Generating high-resolution video streams incurs quadratic computational complexity over spatio-temporal token sequences, leading to severe memory and latency bottlenecks [4][20]. To address this, modern frameworks rely heavily on latent space compression via specialized video variational autoencoders (Video-VAEs) [6][21]. 

For instance, LTX-Video integrates a Video-VAE backbone with transformer-based latent diffusion to achieve realtime video generation while preserving temporal consistency [6]. Similarly, CV-VAE provides compatible video compression that integrates seamlessly with existing image VAEs [22], and KATok introduces a transformer-based VAE with an adaptive token selector to handle varying spatio-temporal complexity rather than fixed compression ratios [5].

To accelerate transformer inference, recent attention mechanisms optimize resource allocation. Heterogeneous attention models like HetA-DiT employ lightweight uncertainty branches to predict token-wise denoising difficulty and adaptively route computation to critical tokens [4]. Other approaches, such as parameterized stripe attention and distillation strategies (e.g., Next-Frame Diffusion at 30+ FPS and Mask Forcing for dual-noise rollout distillation), successfully mitigate inference overhead and quality degradation during self-rollout [23][24][25].

## State-of-the-Art Models and Multimodal Integration
The state of the art in video generation spans high-impact commercial ecosystems and rapidly maturing open-source initiatives [7][8][9][26]. Open-Sora and Open-Sora 2.0 democratize efficient video production by coupling Spatial-Temporal Diffusion Transformers with 3D autoencoders, achieving commercial-grade quality at drastically reduced training costs [27][7]. CogVideoX similarly employs expert transformer backbones and 3D VAEs for robust text-to-video synthesis [8].

Beyond pure text-to-video, multimodal integration has emerged as a defining capability. Media foundation models like Movie Gen synthesize high-quality video alongside perfectly synchronized audio across diverse media tasks [9]. Multimodal reasoning frameworks such as ThinkSound incorporate Chain-of-Thought reasoning within multimodal large language models to govern audio generation and editing alongside video workflows [28]. Comprehensive comparative analyses of contemporary engines indicate that modern generative systems increasingly support native audio generation, fine-grained cinematic camera control, and natural language editing capabilities (such as instruction-based modification and video-to-video transformations) [26].

## Evaluation, Benchmarks, and Post-Training Alignment
Evaluating generative video models presents unique challenges due to the complexity of human perceptual preferences, temporal dynamics, and physical plausibility [10][11]. Generative AI Video Evaluation (AIGVE) frameworks categorize assessment methodologies into metric-based, human-involved, and model-centered paradigms [10]. Standardized benchmarks such as VBench, AIGCBench, and AIGVE-60K systematically assess motion smoothness, temporal consistency, control-video alignment, and overall visual quality [10]. Specialized suites like DynamicEval are specifically designed to uncover and rectify low-dynamic bias in video models [10].

In parallel, post-training and alignment remain vital for ensuring that video generators faithfully follow human intent and respect physical constraints [12]. Pretrained models frequently suffer from error accumulation over long horizons, motion-appearance decoupling, and multi-objective trade-offs [12]. Addressing these challenges requires robust alignment algorithms and efficient world-model paradigms capable of supporting interactive downstream applications such as autonomous driving, robotics, and immersive simulation [15].

## References
[1] A Survey on Video Diffusion Models. hf-search. https://huggingface.co/papers/2310.10647 (2023-10-16)
[2] From Sora What We Can See: A Survey of Text-to-Video Generation. web. https://arxiv.org/html/2405.10674v1 (2024-05-20)
[3] General: Bridging Text and Video Generation: A Survey. arxiv. https://arxiv.org/abs/2510.04999 (2025-10-06)
[4] Where Compute Matters: Heterogeneous Attention for Efficient Video Diffusion. arxiv. https://arxiv.org/abs/2609.31050 (2026-09-25)
[5] Keep-or-Drop? Adaptive Tokenizer for Compact Video Representation. arxiv. https://arxiv.org/abs/2608.24293 (2026-08-25)
[6] LTX-Video: Realtime Video Latent Diffusion. hf-search. https://huggingface.co/papers/2501.00103 (2024-12-30)
[7] Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k. hf-search. https://huggingface.co/papers/2503.09642 (2025-03-12)
[8] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. hf-search. https://huggingface.co/papers/2408.06072 (2024-08-12)
[9] Movie Gen: A Cast of Media Foundation Models. hf-search. https://huggingface.co/papers/2410.13720 (2024-10-17)
[10] Generative AI Video Evaluation: Survey of Metrics, Benchmarks, and Trustworthiness. web. https://openaccess.thecvf.com/content/CVPR2026W/VGBE/papers/Safavigerdini_Generative_AI_Video_Evaluation_Survey_of_Metrics_Benchmarks_and_Trustworthiness_CVPRW_2026_paper.pdf (2026-06-01)
[11] A Survey of AI-Generated Video Evaluation. arxiv. https://arxiv.org/abs/2410.19884 (2026-03-19)
[12] Video Generation Models: A Survey of Post-Training and Alignment. arxiv. https://arxiv.org/abs/2610.00812 (2026-09-30)
[13] Audio-Visual Intelligence in Large Foundation Models. arxiv. https://arxiv.org/abs/2605.04045 (2026-05-05)
[14] Autoregressive Video Generation without Vector Quantization. hf-search. https://huggingface.co/papers/2412.14169 (2024-12-18)
[15] Video Generation Models as World Models: Efficient Paradigms, Architectures and Algorithms. arxiv. https://arxiv.org/abs/2603.28489 (2026-07-04)
[16] VideoAR: Autoregressive Video Generation via Next-Scale Prediction. hf-search. https://huggingface.co/papers/2601.05966 (2026-01-09)
[17] ARLON: Boosting Diffusion Transformers with Autoregressive Models for Long Video Generation. arxiv. https://arxiv.org/abs/2410.20502 (2024-10-27)
[18] AlignMamba: Enhancing Multimodal Mamba with Local and Global Cross-modal Alignment. hf-search. https://huggingface.co/papers/2412.00833 (2024-12-01)
[19] Seeing the Image: Prioritizing Visual Correlation by Contrastive Alignment. hf-search. https://huggingface.co/papers/2405.17871 (2024-05-28)
[20] Efficient Video Diffusion Models: Advancements and Challenges. arxiv. https://arxiv.org/abs/2604.15911 (2026-04-15)
[21] xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations. hf-search. https://huggingface.co/papers/2408.12590 (2024-08-22)
[22] CV-VAE: A Compatible Video VAE for Latent Generative Video Models. hf-search. https://huggingface.co/papers/2405.20279 (2024-05-30)
[23] Playing with Transformer at 30+ FPS via Next-Frame Diffusion. hf-search. https://huggingface.co/papers/2506.01380 (2025-06-02)
[24] Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout. hf-search. https://huggingface.co/papers/2609.09123 (2026-09-08)
[25] Parameterized Stripe Attention for Efficient Video Generation. arxiv. https://arxiv.org/abs/2609.37001 (2026-09-29)
[26] Sora 2 vs Veo 3 vs Runway Gen-3: 2025 AI Video Model Comparison Guide. web. https://skywork.ai/blog/sora-2-vs-veo-3-vs-runway-gen-3-2025-ai-video-generator-comparison/ (2025-10-02)
[27] Open-Sora: Democratizing Efficient Video Production for All. hf-search. https://huggingface.co/papers/2412.20404 (2024-12-29)
[28] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing. hf-search. https://huggingface.co/papers/2506.21448 (2025-06-26)
