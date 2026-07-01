# Official Reference Uncovered Changed Functions

Coverage basis: CPU-only and GPU-only handwritten evaluator traces unioned for `official_reference`.

Summary: 46/255 changed functions covered; 209 uncovered.

| category | uncovered functions | why not covered |
| --- | ---: | --- |
| Optional quantization / QAT / FP8 / ModelOpt | 71 | Current CPU/GPU evaluator uses an unquantized tiny Qwen2 model. Optional quantized/QAT env vars are unset, so FP8, compressed-tensors NVFP4, and ModelOpt NVFP4 reload paths are not entered. |
| Alternative checkpoint / distributed weight-transfer engines | 44 | The current tests exercise standard local vLLM weight updates and some bucket setup, but not KIMI/HCCL/Mooncake engines, large direct CUDA IPC transfer, shared-memory fallback, or explicit stateless distributed process-group flows. |
| vLLM server launch, CLI, config, headless and branch-specific lifecycle | 43 | The tests create servers through the manager path and cover generation/update/lifecycle behavior, but do not exercise every CLI construction branch, non-master headless server branch, quantization branch, MTP drafter branch, or internal helper override. |
| Repository diagnostics / environment checks | 17 | These functions are repo diagnostic utilities or import/setup checks, not the rollout backend behavior under evaluation. |
| NPU / Ascend-specific vLLM patching | 15 | The verified runtime is CUDA. These branches require Ascend/NPU/vllm-ascend runtime conditions and are intentionally inactive here. |
| LoRA/FSDP/Megatron/HF/SGLang side paths | 9 | The task is vLLM backend support; these changed helpers are adjacent compatibility code for other engines or training wrappers and are not triggered by current vLLM CPU/GPU behavior tests. |
| Training / trainer integration helpers outside rollout smoke path | 8 | Current evaluator focuses on rollout behavior. It does not run full PPO/GRPO training, rollout correction, generation-server CLI, teacher/reward loops, or config validation failure paths. |
| Miscellaneous small changed helpers | 2 | Small changed helpers not reached by the current behavior scenarios. |

## Optional quantization / QAT / FP8 / ModelOpt

Current CPU/GPU evaluator uses an unquantized tiny Qwen2 model. Optional quantized/QAT env vars are unset, so FP8, compressed-tensors NVFP4, and ModelOpt NVFP4 reload paths are not entered.

### `verl/utils/modelopt/vllm_modelopt_patch.py` (26)
- `26:_save_param_meta`
- `49:_create_param_from_meta`
- `80:_check_first_call`
- `86:_save_weight_loaders`
- `95:_update_ref_or_create`
- `106:_unwrap_marlin_scale`
- `110:_split_marlin_scale`
- `117:__init__`
- `134:_build_mappings`
- `155:_try_rebuild`
- `175:prepare_for_reload`
- `185:__getitem__`
- `194:__contains__`
- `205:get`
- `215:_modelopt_dense_process_weights`
- `307:_marlin_repack_experts`
- `325:_marlin_process_scales_experts`
- `349:_modelopt_moe_marlin_convert`
- `406:_modelopt_moe_process_weights`
- `445:_modelopt_kv_process_weights`
- `514:_require_fp4_marlin_supported`
- `521:_modelopt_dense_init_marlin`
- `527:_modelopt_moe_init_marlin`
- `543:prepare_modelopt_for_weight_reload`
- `573:modelopt_process_weights_after_loading`
- `598:apply_modelopt_nvfp4_patches`

### `verl/utils/vllm/vllm_fp8_utils.py` (25)
- `49:is_fp8_model`
- `61:get_module_from_param_name`
- `91:is_fp8_weight`
- `108:is_mxfp8_vllm_ascend`
- `122:restore_mxfp8_weights_for_loading`
- `133:apply_mxfp8_transformation_after_loading`
- `157:quant_weights`
- `217:load_quanted_weights`
- `261:_copy_param_subclass_attrs`
- `279:replace_parameter_preserve_subclass`
- `293:_restore_layer_param_subclass_attrs`
- `300:_make_process_weights_after_loading_for_vllm20`
- `301:_patched_process_weights_after_loading`
- `312:process_weights_after_loading_for_vllm10`
- `328:_create_param_from_subclass_attributes`
- `367:process_weights_after_loading_for_vllm11`
- `386:_create_param_from_subclass_attributes`
- `429:process_weights_after_loading_for_vllm14`
- `448:_create_param_from_subclass_attributes`
- `491:process_weights_after_loading_moe_for_vllm10`
- `516:_create_param_from_subclass_attributes`
- `567:process_weights_after_loading_moe_for_vllm11`
- `625:process_weights_after_loading_moe_for_vllm14`
- `653:_create_param_from_subclass_attributes`
- `701:apply_vllm_fp8_patches`

### `verl/utils/qat/vllm_patch.py` (20)
- `48:__init__`
- `75:_build_mappings`
- `113:_try_rebuild`
- `153:prepare_for_reload`
- `163:__getitem__`
- `177:__contains__`
- `193:get`
- `201:_create_param_from_meta`
- `242:save_param_meta`
- `271:_check_first_call`
- `279:patched_w4a16_process_weights_after_loading`
- `369:patched_w4a4_process_weights_after_loading`
- `454:_marlin_repack_experts`
- `474:_marlin_process_scales_experts`
- `495:_process_nvfp4_moe_marlin`
- `586:_process_nvfp4_moe_flashinfer_cutlass`
- `681:patched_nvfp4_moe_process_weights_after_loading`
- `737:apply_qat_patches`
- `756:prepare_qat_for_load_weights`
- `797:manual_process_weights_after_loading`

## Alternative checkpoint / distributed weight-transfer engines

The current tests exercise standard local vLLM weight updates and some bucket setup, but not KIMI/HCCL/Mooncake engines, large direct CUDA IPC transfer, shared-memory fallback, or explicit stateless distributed process-group flows.

### `verl/checkpoint_engine/kimi_checkpoint_engine.py` (13)
- `37:ckpt_get_named_tensor_buckets`
- `66:receive_tensor`
- `193:__init__`
- `208:_run`
- `212:wait_for_complete`
- `234:__init__`
- `248:prepare`
- `259:finalize`
- `268:build_topology`
- `285:init_process_group`
- `321:send_weights`
- `332:offload_cpu`
- `366:receive_weights`

### `verl/checkpoint_engine/hccl_checkpoint_engine.py` (12)
- `57:__init__`
- `75:_run`
- `87:wait_for_complete`
- `109:__init__`
- `131:prepare`
- `141:finalize`
- `155:build_topology`
- `168:_start_zmq_server`
- `182:_connect_zmq_client`
- `195:init_process_group`
- `230:send_weights`
- `307:receive_weights`

### `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` (9)
- `45:rebuild_ipc`
- `56:create_shared_memory`
- `66:rebuild_shared_memory`
- `216:_direct_send_large_weight`
- `246:__init__`
- `261:receive_weights`
- `296:_init_socket`
- `301:_init_buffer`
- `317:_cleanup`

### `verl/checkpoint_engine/mooncake_checkpoint_engine.py` (8)
- `45:__init__`
- `88:prepare`
- `98:build_topology`
- `111:init_process_group`
- `135:finalize`
- `142:wait_for_complete`
- `150:send_weights`
- `226:receive_weights`

### `verl/utils/distributed.py` (2)
- `99:stateless_init_process_group`
- `122:create_process_group`

## vLLM server launch, CLI, config, headless and branch-specific lifecycle

The tests create servers through the manager path and cover generation/update/lifecycle behavior, but do not exercise every CLI construction branch, non-master headless server branch, quantization branch, MTP drafter branch, or internal helper override.

### `verl/workers/rollout/vllm_rollout/vllm_async_server.py` (19)
- `91:__init__`
- `205:launch_server`
- `385:run_server`
- `433:run_headless`
- `437:run_headless_wrapper`
- `441:on_run_headless_done`
- `644:release_kv_cache`
- `651:resume_kv_cache`
- `802:_init_config`
- `806:_init_model_config`
- `810:_validate_configs`
- `822:_post_init`
- `831:_get_engine_kwargs_key`
- `835:_preprocess_engine_kwargs`
- `839:_get_override_generation_config`
- `851:_apply_quantization`
- `916:_get_worker_extension_cls`
- `920:_get_cli_modules`
- `924:_get_cli_description`

### `verl/workers/rollout/vllm_rollout/utils.py` (18)
- `43:set_death_signal`
- `68:get_vllm_max_lora_rank`
- `91:monkey_patch_compute_logits`
- `94:compute_logits`
- `120:__new__`
- `157:_get_drafter_model`
- `162:_get_draft_model_config`
- `167:_use_mtp_drafter_weight_sync`
- `172:_iter_all_models`
- `182:_iter_all_models_with_config`
- `190:monkey_patch_model`
- `197:update_weights_from_ipc`
- `262:_update_weights`
- `290:_get_zmq_handle`
- `305:__enter__`
- `308:no_op_signal`
- `317:__exit__`
- `321:build_cli_args_from_config`

### `verl/utils/vllm/utils.py` (4)
- `38:hijack`
- `39:hijack__load_adapter`
- `120:do_hijack`
- `126:is_version_ge`

### `verl/utils/vllm/patch.py` (1)
- `77:patch_vllm_moe_model_weight_loader`

### `verl/workers/rollout/vllm_rollout/vllm_rollout.py` (1)
- `51:_check_vllm_version_for_sleep_level`

## Repository diagnostics / environment checks

These functions are repo diagnostic utilities or import/setup checks, not the rollout backend behavior under evaluation.

### `scripts/diagnose.py` (16)
- `50:test_connection`
- `70:check_python`
- `78:check_pip`
- `89:_get_current_git_commit`
- `101:check_verl`
- `126:check_os`
- `135:check_hardware`
- `151:check_network`
- `170:check_environment`
- `177:check_pip_package_versions`
- `187:check_cuda_versions`
- `208:_get_cpu_memory`
- `216:_get_gpu_info`
- `244:_get_system_info`
- `253:check_system_info`
- `263:parse_args`

### `verl/utils/import_utils.py` (1)
- `37:is_vllm_available`

## NPU / Ascend-specific vLLM patching

The verified runtime is CUDA. These branches require Ascend/NPU/vllm-ascend runtime conditions and are intentionally inactive here.

### `verl/utils/vllm/npu_vllm_patch.py` (15)
- `24:vllm_ascend_v011_select_moe_comm_method_wrapper`
- `26:wrapper`
- `53:vllm_ascend_v011_matmul_and_reduce_wrapper`
- `55:wrapper`
- `74:check_vllm_ascend_before_server_launch`
- `78:_is_ascend_soc_version_A2_v011_local`
- `91:_is_ascend_soc_version_A2_v013_local`
- `124:vllm_ascend_v013_select_moe_comm_method_wrapper`
- `126:wrapper`
- `142:vllm_ascend_v013_matmul_and_reduce_wrapper`
- `144:wrapper`
- `163:vllm_v013_weight_loader_method_wrapper`
- `165:wrapper`
- `175:patch_vllm013_rotary_emb`
- `178:vllm013_npu_rotary_embedding_init_impl`

## LoRA/FSDP/Megatron/HF/SGLang side paths

The task is vLLM backend support; these changed helpers are adjacent compatibility code for other engines or training wrappers and are not triggered by current vLLM CPU/GPU behavior tests.

### `verl/utils/fsdp_utils.py` (3)
- `673:collect_lora_params`
- `731:replace_lora_wrapper`
- `1009:merged_lora_context`

### `verl/workers/rollout/utils.py` (2)
- `92:qwen2_5_vl_dedup_image_tokens`
- `116:update_prometheus_config`

### `verl/utils/megatron_peft_utils.py` (1)
- `131:build_peft_config_for_vllm`

### `verl/workers/engine/megatron/transformer_impl.py` (1)
- `720:get_per_tensor_param`

### `verl/workers/rollout/hf_rollout.py` (1)
- `54:_generate_minibatch`

### `verl/workers/rollout/sglang_rollout/async_sglang_server.py` (1)
- `61:_extract_prompt_logprobs_sglang`

## Training / trainer integration helpers outside rollout smoke path

Current evaluator focuses on rollout behavior. It does not run full PPO/GRPO training, rollout correction, generation-server CLI, teacher/reward loops, or config validation failure paths.

### `verl/trainer/ppo/rollout_corr_helper.py` (2)
- `779:compute_rollout_correction_and_rejection_mask`
- `897:compute_offpolicy_metrics`

### `verl/experimental/reward_loop/reward_loop.py` (1)
- `231:compute_score_disrm`

### `verl/experimental/teacher_loop/teacher_manager.py` (1)
- `30:_get_teacher_sampling_params`

### `verl/trainer/main_generation_server.py` (1)
- `123:main`

### `verl/trainer/ppo/core_algos.py` (1)
- `2271:compute_policy_loss_reinforce`

### `verl/trainer/ppo/ray_trainer.py` (1)
- `136:compute_spec_decode_metrics`

### `verl/utils/config.py` (1)
- `74:validate_config`

## Miscellaneous small changed helpers

Small changed helpers not reached by the current behavior scenarios.

### `verl/utils/profiler/config.py` (1)
- `191:build_vllm_profiler_args`

### `verl/workers/config/distillation.py` (1)
- `176:_validate_topk_logprobs`
