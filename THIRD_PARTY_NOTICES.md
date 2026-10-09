# Third-party code notices

OCS-LDA includes modified and inlined Swin-family source code. The repository-level `LICENSE` is the Apache License 2.0 for the authors' licensable contributions; it does **not** erase the copyright or licence notices applicable to third-party portions. The original source projects and their licence texts are identified below. The cited commit hashes fix versions for provenance comparison; they do not establish the exact historical import commits.

| Project | Origin and licence | Local files and changes |
|---|---|---|
| SwinIR, by Jingyun Liang and contributors | [Source](https://github.com/JingyunLiang/SwinIR/tree/6545850fbf8df298df73d81f3e8cba638787c8bd); [Apache-2.0 licence](https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/LICENSE). The full Apache-2.0 text is also the repository's `LICENSE`. | `models/SwimIRv2.py` adapts SwinIR building blocks for regional atmospheric fields and observation processing. `models/ocs_lda_blocks.py` inlines and extends related blocks for the structured OCS-LDA autoencoder. These are modified files, not verbatim upstream releases. |
| Swin Transformer, Copyright (c) Microsoft Corporation | [Source](https://github.com/microsoft/Swin-Transformer/tree/f82860bfb5225915aca09c3227159ee9e1df874d); MIT. Full notice: [`THIRD_PARTY_LICENSES/SWIN_TRANSFORMER_MIT.txt`](THIRD_PARTY_LICENSES/SWIN_TRANSFORMER_MIT.txt). | Swin block implementations in the above two local files share continuous source-code passages with the upstream implementation; some of this overlap is inherited through SwinIR. |
| KAIR, Copyright (c) 2019 Kai Zhang | [Source](https://github.com/cszn/KAIR/tree/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e); MIT. Full notice: [`THIRD_PARTY_LICENSES/KAIR_MIT.txt`](THIRD_PARTY_LICENSES/KAIR_MIT.txt). | KAIR's `network_swinir.py` shares substantial SwinIR-family code with the above local files. SwinIR's own acknowledgement directs downstream users to follow KAIR's licence. |

The cited projects are credited for their prior implementations. The OCS-LDA authors' contributions include the atmospheric-domain model integration, Shared–Private representation, observation-conditioned latent updating and assimilation workflow. This notice does not imply endorsement by the upstream authors.

The external [LDA_1.41 repository](https://github.com/hangfan99/LDA_1.41) is used as a research comparator but is **not** bundled or relicensed in this code release. Cite the LDA work as specified in the manuscript. As checked on 2026-10-08, the upstream GitHub repository did not expose a `LICENSE` file; do not infer open-source reuse permission from public visibility alone.

