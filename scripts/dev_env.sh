#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Development environment for dynG. SOURCE this file (do not execute it):
#
#   source scripts/dev_env.sh
#
# It activates the conda environment 'dyng-dev' (created from environment.yml) or, if conda
# cannot be initialised, puts that environment's bin/ on PATH; adds the CUDA toolkit's bin/ to
# PATH when it exists (nvcc is not on PATH by default on the development machine); and exports
# DYNG_SCRATCH, the persistent work area of the parity harness.
#
# Overridable through the environment:
#   DYNG_CONDA_ENV   conda environment name       (default: dyng-dev)
#   DYNG_CONDA_ROOT  conda installation prefix    (default: $CONDA_PREFIX of base, ~/anaconda3, ~/miniconda3)
#   DYNG_CUDA_HOME   CUDA toolkit prefix          (default: /usr/local/cuda-13.1, else /usr/local/cuda)
#   DYNG_SCRATCH     parity/benchmark work area   (default: $HOME/Projects/dyng-work)

if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "dev_env.sh must be sourced: source scripts/dev_env.sh" >&2
  exit 1
fi

_dyng_env="${DYNG_CONDA_ENV:-dyng-dev}"
_dyng_conda_root="${DYNG_CONDA_ROOT:-}"
if [ -z "${_dyng_conda_root}" ]; then
  for _dyng_candidate in "${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE}")")}" \
    "${HOME}/anaconda3" "${HOME}/miniconda3" "${HOME}/miniforge3"; do
    if [ -n "${_dyng_candidate}" ] && [ -f "${_dyng_candidate}/etc/profile.d/conda.sh" ]; then
      _dyng_conda_root="${_dyng_candidate}"
      break
    fi
  done
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "${_dyng_env}" ]; then
  if [ -n "${_dyng_conda_root}" ] && [ -f "${_dyng_conda_root}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "${_dyng_conda_root}/etc/profile.d/conda.sh"
    if ! conda activate "${_dyng_env}" 2>/dev/null; then
      if [ -d "${_dyng_conda_root}/envs/${_dyng_env}/bin" ]; then
        export PATH="${_dyng_conda_root}/envs/${_dyng_env}/bin:${PATH}"
      else
        echo "dev_env.sh: conda env '${_dyng_env}' not found; run: conda env create -f environment.yml" >&2
      fi
    fi
  else
    echo "dev_env.sh: conda not found; make sure cmake (>= 3.30) and ninja are on PATH" >&2
  fi
fi

_dyng_cuda="${DYNG_CUDA_HOME:-}"
if [ -z "${_dyng_cuda}" ]; then
  if [ -x /usr/local/cuda-13.1/bin/nvcc ]; then
    _dyng_cuda=/usr/local/cuda-13.1
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    _dyng_cuda=/usr/local/cuda
  fi
fi
if [ -n "${_dyng_cuda}" ] && [ -d "${_dyng_cuda}/bin" ]; then
  case ":${PATH}:" in
    *":${_dyng_cuda}/bin:"*) ;;
    *) export PATH="${PATH}:${_dyng_cuda}/bin" ;;
  esac
  export CUDACXX="${_dyng_cuda}/bin/nvcc"
fi

export DYNG_SCRATCH="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
# Share one CPM download cache between build trees (keeps build directories small).
export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-${DYNG_SCRATCH}/cpm-cache}"

unset _dyng_env _dyng_conda_root _dyng_candidate _dyng_cuda
