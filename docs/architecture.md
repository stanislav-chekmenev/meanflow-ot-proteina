# MeanFlow-OT-Proteina: Architecture & Data Flow

## 1. High-Level System Overview

```mermaid
flowchart TB
    subgraph DATA["Data Pipeline"]
        PDB["PDB .cif files"]
        PyG["protein_to_pyg()<br/>coords [N,37,3] A"]
        DS["PDBDataset<br/>+ Transforms"]
        DL["DensePaddingDataLoader<br/>coords [B, N_pad, 37, 3]"]
        PDB --> PyG --> DS --> DL
    end

    subgraph TRAIN["Training Loop (ModelTrainerBase)"]
        EX["extract_clean_sample<br/>CA atoms -> x_1 [B,N,3] nm"]
        OT["OT Coupling<br/>(optional)"]
        INTERP["Interpolation<br/>z = (1-t)*x_1 + t*x_0"]
        JVP["MeanFlow JVP<br/>torch.func.jvp"]
        LOSS["Loss Computation<br/>FM + MeanFlow + Adaptive Wt"]

        EX --> OT --> INTERP --> JVP --> LOSS
    end

    subgraph MODEL["ProteinTransformerAF3"]
        FEAT["Feature Factories"]
        TRUNK["Transformer Trunk<br/>(N layers)"]
        DEC["Coordinate Decoder<br/>[B,N,3]"]
        FEAT --> TRUNK --> DEC
    end

    subgraph INFER["Inference"]
        NOISE["z_1 ~ N(0,I)<br/>[B,N,3]"]
        PRED["u = nn(z_1, t=1, h=1)"]
        OUT["z_0 = z_1 - u<br/>-> atom37 [B,N,37,3] A"]
        NOISE --> PRED --> OUT
    end

    DL --> EX
    JVP --> MODEL
    MODEL --> JVP
    NOISE --> MODEL
    MODEL --> PRED
```

## 2. ProteinTransformerAF3 — Neural Network Architecture

```mermaid
flowchart TB
    subgraph INPUTS["Inputs"]
        XT["x_t [B,N,3]<br/>noisy CA coords (nm)"]
        T["t [B]<br/>timestep"]
        H["h = t-r [B]<br/>MeanFlow interval"]
        MASK["mask [B,N]<br/>residue mask"]
        CATH["cath_code<br/>fold labels"]
        XSC["x_sc [B,N,3]<br/>self-conditioning"]
    end

    subgraph COND["Conditioning Vector c [B,N,512]"]
        TE["TimeEmb<br/>sin/cos(t) [B,N,256]"]
        DE["DeltaTEmb<br/>sin/cos(h) [B,N,256]"]
        FE["FoldEmb<br/>CATH C.A.T [B,N,768]"]
        CPROJ["Linear -> 512<br/>+ 2x SwiGLU Transition"]
        TE & DE & FE --> CPROJ
    end

    subgraph SEQ_INIT["Initial Sequence Repr [B,N,512]"]
        IDX["IdxEmb<br/>sin/cos(pdb_idx) [B,N,128]"]
        CB["ChainBreak<br/>[B,N,1]"]
        SC["x_sc<br/>[B,N,3]"]
        SPROJ["concat -> Linear -> 512"]
        COORD_E["Linear(3 -> 512)<br/>x_t embedding"]
        ADD_S["+ (element-wise add)"]
        IDX & CB & SC --> SPROJ
        SPROJ --> ADD_S
        COORD_E --> ADD_S
    end

    subgraph PAIR["Pair Repr [B,N,N,256]"]
        SEP["SeqSep<br/>rel. position [B,N,N,127]"]
        PD_SC["x_sc dists<br/>binned [B,N,N,128]"]
        PD_XT["x_t dists<br/>binned [B,N,N,64]"]
        PPROJ["concat -> Linear -> 256<br/>(+ AdaLN with t,h)"]
        SEP & PD_SC & PD_XT --> PPROJ
    end

    subgraph REG["Register Tokens"]
        RTOK["10 learnable tokens<br/>prepended to seq dim"]
    end

    subgraph TRUNK["Transformer Trunk (10 layers)"]
        direction TB
        L1["Layer 1"]
        L2["Layer 2"]
        LN["..."]
        L10["Layer 10"]
        L1 --> L2 --> LN --> L10
    end

    subgraph LAYER["Each Transformer Layer"]
        direction TB
        ADALN1["AdaLN(seqs, c)"]
        ATTN["PairBiasAttention<br/>Q,K,V + pair bias<br/>8 heads, QK LayerNorm<br/>sigmoid gating"]
        SCALE1["AdaptiveOutputScale(c)"]
        RES1["+ residual"]
        ADALN2["AdaLN(seqs, c)"]
        FFN["SwiGLU Transition<br/>512 -> 2048 -> 512"]
        SCALE2["AdaptiveOutputScale(c)"]
        RES2["+ residual"]

        ADALN1 --> ATTN --> SCALE1 --> RES1
        RES1 --> ADALN2 --> FFN --> SCALE2 --> RES2
    end

    subgraph OUTPUT["Output"]
        STRIP["Strip register tokens"]
        LN_OUT["LayerNorm"]
        LIN_OUT["Linear(512 -> 3)"]
        COORS["coors_pred [B,N,3]"]
        STRIP --> LN_OUT --> LIN_OUT --> COORS
    end

    T --> TE
    H --> DE
    CATH --> FE
    XT --> COORD_E
    XSC --> SC
    XSC --> PD_SC
    XT --> PD_XT

    CPROJ --> COND
    ADD_S --> REG
    PPROJ --> PAIR

    REG --> TRUNK
    PAIR -.->|"bias"| TRUNK
    COND -.->|"AdaLN"| TRUNK

    TRUNK --> OUTPUT
```

## 3. Training Step — MeanFlow with OT Coupling

```mermaid
flowchart TB
    subgraph STEP1["1. Data Preparation"]
        BATCH["batch from DataLoader"]
        X1["x_1 = CA atoms [B,N,3]<br/>A -> nm, zero CoM"]
        BATCH --> X1
    end

    subgraph STEP2["2. OT Coupling (optional)"]
        direction TB
        X0_RAW["x_0 ~ N(0,I) [B,N,3]<br/>reference noise"]

        subgraph OT_MODES["Coupling Mode"]
            NONE["No OT<br/>x_0 used as-is"]
            MINI["Mini-batch OT<br/>Hungarian on [B,B]<br/>cost matrix"]
            SQUARE["Square OT Pool<br/>K proteins, K noise<br/>Hungarian on [K,K]<br/>select B pairs"]
        end

        X0_RAW --> OT_MODES
        OT_MODES --> X0["x_0 [B,N,3]<br/>(coupled noise)"]
    end

    subgraph STEP3["3. Time Sampling"]
        TS["Sample t ~ logit-normal<br/>P_mean=-0.4, P_std=1.0"]
        RS["Sample r:<br/>with prob 0.75: r = t<br/>with prob 0.25: r ~ logit-normal<br/>r = min(t, r)"]
        TS --> RS
    end

    subgraph STEP4["4. Interpolation"]
        Z["z = (1-t)*x_1 + t*x_0<br/>noisy sample at time t"]
        V["v = x_0 - x_1<br/>conditional velocity"]
    end

    subgraph STEP5["5. MeanFlow Loss (JVP)"]
        direction TB
        UFUNC["u(z, t, r) = nn(z, t, h=t-r)"]
        JVPOP["torch.func.jvp(<br/>u_func,<br/>primals=(z, t, r),<br/>tangents=(v, 1, 0))"]
        UPRED["u_pred = u(z_t, r, t)"]
        DUDT["dudt = du/dt"]
        UTGT["u_tgt = (v - (t-r)*dudt).detach()"]
        LMFN["loss_mf = ||u_pred - u_tgt||^2 / nres"]

        UFUNC --> JVPOP
        JVPOP --> UPRED & DUDT
        DUDT --> UTGT
        UPRED & UTGT --> LMFN
    end

    subgraph STEP6["6. FM Loss (r=t subset)"]
        VPRED["v_pred = nn(z, t, h=0)"]
        LFM["loss_fm = ||v_pred - v||^2 / nres"]
        VPRED --> LFM
    end

    subgraph STEP7["7. Combined Loss"]
        AW["Adaptive weight:<br/>w = (loss + 0.001)^1.0"]
        COMB["combined = 0.75*loss_fm/w<br/>+ 0.25*loss_mf/w"]
        AW --> COMB
    end

    X1 --> STEP2
    STEP2 --> STEP4
    STEP3 --> STEP4
    STEP4 --> STEP5
    STEP4 --> STEP6
    STEP5 --> STEP7
    STEP6 --> STEP7
```

## 4. Inference — MeanFlow Sampling

```mermaid
flowchart LR
    subgraph ONESTEP["1-Step Sampling (default)"]
        Z1["z_1 ~ N(0,I)<br/>[B,N,3]"]
        U1["u = nn(z_1, t=1, h=1)"]
        Z0["z_0 = z_1 - u"]
        Z1 --> U1 --> Z0
    end

    subgraph MULTI["Multi-Step Sampling"]
        ZC["z = z_1"]
        LOOP["for t_cur, t_next<br/>in linspace(1,0,S+1):"]
        UM["u = nn(z, t_cur,<br/>h=t_cur-t_next)"]
        ZN["z = z - (t_cur-t_next)*u"]
        ZC --> LOOP --> UM --> ZN
        ZN -->|"next step"| LOOP
    end

    subgraph POST["Post-processing"]
        NM["z_0 [B,N,3] nm"]
        A37["atom37 [B,N,37,3] A<br/>(CA at index 1)"]
        NM --> A37
    end

    Z0 --> NM
    ZN --> NM
```

## 5. Model Variants

| Config | Params | token_dim | layers | heads | pair_dim | Triangle Mult | Pair Update |
|--------|--------|-----------|--------|-------|----------|---------------|-------------|
| `ca_af3_60M_notri` | ~60M | 512 | 10 | 8 | 256 | No | No |
| `ca_af3_200M_no_tri` | ~200M | 768 | 15 | 12 | 512 | No | Every 3 layers |
| `ca_af3_200M_yes_tri` | ~200M | 768 | 15 | 12 | 512 | Yes | Every 3 layers |
| `ca_af3_400M_yes_tri` | ~400M | 1024 | 18 | 16 | 512 | Yes | Every 5 layers |

All variants use 10 register tokens. Note: MeanFlow training requires `update_pair_repr=False` due to JVP incompatibility with `torch.utils.checkpoint`.

## 6. Key Design Decisions

- **Coordinate space**: All internal computation in nanometers; PDB data in Angstroms; conversion at boundaries
- **Center-of-mass**: Zeroed throughout training and inference via `_mask_and_zero_com`
- **Time convention**: t=1 is noise, t=0 is data (opposite of some flow matching papers)
- **MeanFlow**: Network predicts *average velocity* u(z_t, r, t) where h=t-r is passed as conditioning. When r=t (75% of training), collapses to standard flow matching
- **OT coupling**: Hungarian algorithm (deterministic) on masked squared-Euclidean cost; optional square pool samples extra proteins for better coupling quality
- **Adaptive loss weighting**: Per-sample normalization by loss magnitude (Eq. 22) prevents large-loss samples from dominating
- **Self-conditioning**: Previous prediction's coordinates fed back as input features (x_sc); zeros on first pass
- **Fold conditioning**: CATH hierarchy labels with per-level random masking for classifier-free guidance
