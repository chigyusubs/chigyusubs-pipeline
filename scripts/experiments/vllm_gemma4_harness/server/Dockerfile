FROM vllm/vllm-openai-rocm:nightly

# Audio extras for vLLM's multimodal input path (librosa/soundfile/av/resampy/soxr).
# MUST go before the transformers upgrade — `vllm[audio]` pins transformers<5.
RUN pip install --no-cache-dir 'vllm[audio]' \
 && pip install --no-cache-dir --upgrade transformers==5.5.3
