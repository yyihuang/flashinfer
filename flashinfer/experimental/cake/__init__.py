"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Experimental Cake packages without a run-to-run bitwise contract: the
non-deterministic counterparts of the deterministic (``*_det_*``) Cake training
packages.  A package's non-deterministic programs run only in a call made with
``deterministic=False`` or in a continuation of such a call; each family's entry
docstrings and its README or module documentation state how the other values of
``deterministic`` and ``torch.use_deterministic_algorithms(True)`` behave.  Each
subpackage is standalone (import it by its full name); this module imports
nothing.
"""
