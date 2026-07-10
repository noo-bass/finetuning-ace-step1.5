"""ACE-Step 1.5 inference service on Modal: generate audio with a trained LoRA.

Loads the frozen turbo base model, optionally attaches a trained LoRA adapter,
and runs text2music generation -- no 5Hz LM needed since we already have real
caption/lyrics/bpm/key metadata (LM is only required for CoT reasoning modes).

This is a first, minimal version to prove the LoRA-in-the-loop path works end
to end on Modal (see modal/README.md Status section) -- not yet a deployed
multi-tenant inference API. No warm-keeping, no per-user isolation.

Run:
    modal run modal/generate_service.py::generate \\
        --job-id mkgee-e2e-test --track-index 0 --use-lora true
"""

import modal

app = modal.App("acestep-lora-generate")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "torchaudio==2.10.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers>=4.51.0,<4.58.0",
        "diffusers>=0.37.0",
        "accelerate>=1.12.0",
        "peft>=0.18.0",
        "lycoris-lora",
        "soundfile",
        "loguru",
        "einops",
        "numba",
        "vector-quantize-pytorch",
        "torchao>=0.16.0,<0.17.0",
        "toml",
        "pytorch-wavelets",
        "pywavelets",
        "torchcodec>=0.9.1",
    )
    .add_local_dir("../acestep", "/root/acestep", copy=True,
                    ignore=["**/__pycache__", "**/*_test.py"])
)

checkpoints_vol = modal.Volume.from_name("acestep-checkpoints", create_if_missing=True)
jobs_vol = modal.Volume.from_name("acestep-lora-jobs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/checkpoints": checkpoints_vol, "/root/jobs": jobs_vol},
    timeout=600,
)
def generate(job_id: str, track_index: int = 0, use_lora: bool = True, seed: int = 42,
             lora_subpath: str = "lora_output/final", caption_override: str = "",
             save_subdir: str = "", adapter_strength: float = 1.0,
             adapter_mask_steps: int = 0, task_type: str = "text2music",
             src_audio: str = "", reference_audio: str = "",
             flow_edit_morph: bool = False, flow_edit_n_min: float = 0.0,
             flow_edit_n_max: float = 1.0, flow_edit_source_caption: str = "",
             flow_edit_source_lyrics: str = "") -> dict:
    """task_type: text2music | cover | repaint | lego | extract | complete.
    src_audio / reference_audio are job-relative paths on the volume (e.g.
    "audio/Mk.gee - Alesis ... .wav"). flow_edit_morph steers text2music
    with a source track; n_min/n_max bound which portion of the denoising
    trajectory is re-generated -- effectively a transform-strength window
    (low n_max = gentle recolor, full window = heavy reimagining)."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    dataset_json = Path(f"/root/jobs/{job_id}/dataset.json")
    dataset = json.loads(dataset_json.read_text(encoding="utf-8"))
    sample = dict(dataset["samples"][track_index])
    if caption_override:
        # Isolate caption as the only changed variable -- lyrics/bpm/keyscale/
        # timesignature stay from the original dataset entry, so any difference
        # in the generated audio is attributable to the caption alone.
        sample["caption"] = caption_override
    print(f"[generate] Using track: {sample['filename']} (lora={'on' if use_lora else 'off'})")
    if caption_override:
        print(f"[generate] Caption overridden: {caption_override}")

    dit_handler = AceStepHandler()
    init_status, init_ok = dit_handler.initialize_service(
        project_root="/root/checkpoints/checkpoints",
        config_path="acestep-v15-turbo",
        device="cuda",
    )
    print(f"[generate] init_service: {init_status}")
    if not init_ok:
        raise RuntimeError(f"DiT init failed: {init_status}")

    lora_status = None
    lora_verification = None
    mask_state = None
    if use_lora:
        lora_path = f"/root/jobs/{job_id}/{lora_subpath}"
        # add_lora() derives the adapter name from the checkpoint dir's
        # basename when none is given, and torch's nn.Module.add_module
        # rejects any name containing "." -- checkpoint dirs are named
        # like "epoch_200_loss_0.9026" (dot in the loss value), which
        # fails with KeyError: module name can't contain ".". Always pass
        # an explicit, dot-free name.
        lora_status = dit_handler.add_lora(lora_path, adapter_name="loaded_adapter")
        print(f"[generate] LoRA load status: {lora_status}")
        if lora_status.startswith("❌"):
            raise RuntimeError(f"LoRA load failed: {lora_status}")

        # Don't just trust the status string -- this session has already hit
        # multiple "success" messages that meant nothing (silent 0/12
        # preprocessing, 0-second training runs). Actually inspect the model.
        # Two distinct adapter families load via different wrappers:
        # LoRA/DoRA -> PEFT's PeftModel, params named "*lora_*"
        # LoKr/LoHA  -> LyCORIS wrapping (no PeftModel), params named "*lokr_*"/"*loha_*"
        from peft import PeftModel
        decoder = dit_handler.model.decoder
        is_peft = isinstance(decoder, PeftModel)
        has_lycoris = hasattr(decoder, "_lycoris_net") or any(
            "_lycoris_net" in name for name, _ in decoder.named_modules()
        )
        adapter_param_count = 0
        adapter_nonzero_count = 0
        marker = "lora_" if is_peft else ("lokr_" if has_lycoris else None)
        if marker:
            for name, p in decoder.named_parameters():
                if marker in name:
                    adapter_param_count += p.numel()
                    adapter_nonzero_count += int((p != 0).sum().item())
        lora_verification = {
            "adapter_family": "peft_lora" if is_peft else ("lycoris" if has_lycoris else "unknown"),
            "decoder_is_peft_model": is_peft,
            "decoder_has_lycoris": has_lycoris,
            "active_adapters": list(decoder.peft_config.keys()) if is_peft else [],
            "adapter_param_count": adapter_param_count,
            "adapter_nonzero_param_count": adapter_nonzero_count,
        }
        print(f"[generate] LoRA structural verification: {lora_verification}")
        if (not is_peft and not has_lycoris) or adapter_param_count == 0:
            raise RuntimeError(
                f"LoRA reported success but decoder shows no PEFT or LyCORIS "
                f"wrapping, or 0 adapter params: {lora_verification}"
            )

        # Adapter strength: interpolate between base-model behavior (0.0)
        # and full adapter effect (1.0+). ACE's add_lora has no such knob,
        # but PEFT LoraLayers expose per-adapter `scaling` (= alpha/r) we
        # can multiply post-load; LyCORIS networks expose a `multiplier`.
        # An overcooked adapter at strength ~0.5 often keeps its character
        # while restoring the base model's musical prior.
        if adapter_strength != 1.0:
            scaled_layers = 0
            if is_peft:
                for module in decoder.modules():
                    scaling = getattr(module, "scaling", None)
                    if isinstance(scaling, dict) and "loaded_adapter" in scaling:
                        scaling["loaded_adapter"] *= adapter_strength
                        scaled_layers += 1
            elif has_lycoris:
                net = getattr(decoder, "_lycoris_net", None)
                if net is not None and hasattr(net, "multiplier"):
                    net.multiplier = adapter_strength
                    scaled_layers = len(getattr(net, "loras", []) or [])
            if scaled_layers == 0:
                raise RuntimeError(
                    f"adapter_strength={adapter_strength} requested but no "
                    f"scalable adapter layers were found -- refusing to "
                    f"silently generate at full strength"
                )
            print(f"[generate] adapter strength set to {adapter_strength} "
                  f"across {scaled_layers} layers")

        # Step-masked inference (dadabots/T-LoRA): keep the adapter OFF for
        # the first N denoising steps so the frozen base model decides song
        # structure, then switch it on to style the surface. Implemented by
        # wrapping decoder.forward with a call counter -- with turbo's
        # 8-step schedule and no CFG doubling, one decoder call == one step.
        # decoder_calls is reported in the result so that assumption is
        # verified, not trusted.
        if adapter_mask_steps > 0:
            peft_scalings = []
            if is_peft:
                for module in decoder.modules():
                    scaling = getattr(module, "scaling", None)
                    if isinstance(scaling, dict) and "loaded_adapter" in scaling:
                        peft_scalings.append((scaling, scaling["loaded_adapter"]))
            lyc_net = getattr(decoder, "_lycoris_net", None) if has_lycoris else None
            lyc_on = getattr(lyc_net, "multiplier", 1.0) if lyc_net is not None else None
            if not peft_scalings and lyc_net is None:
                raise RuntimeError(
                    f"adapter_mask_steps={adapter_mask_steps} requested but no "
                    f"maskable adapter layers were found"
                )
            mask_state = {"calls": 0}
            orig_forward = decoder.forward

            def masked_forward(*args, **kwargs):
                off = mask_state["calls"] < adapter_mask_steps
                for scaling, on_val in peft_scalings:
                    scaling["loaded_adapter"] = 0.0 if off else on_val
                if lyc_net is not None:
                    lyc_net.multiplier = 0.0 if off else lyc_on
                mask_state["calls"] += 1
                return orig_forward(*args, **kwargs)

            decoder.forward = masked_forward
            print(f"[generate] adapter masked for first {adapter_mask_steps} "
                  f"decoder calls")

    audio_kwargs = {}
    if src_audio:
        audio_kwargs["src_audio"] = f"/root/jobs/{job_id}/{src_audio}"
    if reference_audio:
        audio_kwargs["reference_audio"] = f"/root/jobs/{job_id}/{reference_audio}"
    if flow_edit_morph:
        audio_kwargs.update(
            flow_edit_morph=True,
            flow_edit_n_min=flow_edit_n_min,
            flow_edit_n_max=flow_edit_n_max,
            flow_edit_source_caption=flow_edit_source_caption,
            flow_edit_source_lyrics=flow_edit_source_lyrics,
        )
    if audio_kwargs:
        print(f"[generate] audio conditioning: task={task_type} {audio_kwargs}")

    params = GenerationParams(
        task_type=task_type,
        caption=sample["caption"],
        lyrics=sample["lyrics"],
        instrumental=sample.get("is_instrumental", False),
        vocal_language=sample.get("language") or "unknown",
        bpm=sample.get("bpm"),
        keyscale=sample.get("keyscale") or "",
        timesignature=str(sample.get("timesignature") or ""),
        duration=min(sample.get("duration", 60) or 60, 60),  # cap for a quick sanity listen
        inference_steps=8,   # turbo default
        shift=3.0,           # turbo default
        seed=seed,
        **audio_kwargs,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=seed)

    # Distinct save_subdir per adapter config matters when sweeping: LoKr's
    # weights-hash is always "" (see STATUS.md), so two different LoKr
    # checkpoints generating with the same seed/prompt produce the SAME
    # output filename and would silently overwrite each other in a shared dir.
    save_dir = f"/root/jobs/{job_id}/generations/" + (
        save_subdir or ("lora_on" if use_lora else "lora_off"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    result = generate_music(dit_handler, None, params, config, save_dir=save_dir)

    if mask_state is not None:
        print(f"[generate] decoder was called {mask_state['calls']} times "
              f"({adapter_mask_steps} masked); inference_steps={params.inference_steps}")

    jobs_vol.commit()
    # audios[i] includes a raw torch.Tensor waveform -- fine to pickle leaving
    # the (torch-equipped) container, but fails to *deserialize* back on a
    # local `modal run` client that doesn't have torch installed. Strip it;
    # the actual audio is already written to disk at `path`.
    audio_paths = [{k: v for k, v in a.items() if k != "tensor"} for a in result.audios]
    print(f"[generate] success={result.success} status={result.status_message}")
    print(f"[generate] audios: {audio_paths}")
    if not result.success:
        raise RuntimeError(f"Generation failed: {result.error or result.status_message}")

    return {
        "job_id": job_id,
        "track": sample["filename"],
        "use_lora": use_lora,
        "lora_status": lora_status,
        "lora_verification": lora_verification,
        "save_dir": save_dir,
        "audios": audio_paths,
        "status_message": result.status_message,
    }
