# Contributing

Contributions should preserve the separation between numerical facts,
corpus-bound inference, display policy, and generated interpretation.

1. Open an issue describing the scientific or operational change.
2. Create a focused branch and add tests for changed behavior.
3. Run the Python and web verification commands in the main README.
4. Document every new scientific dependency, version, seed, threshold, and
   invalidation rule.
5. Do not commit source datasets, local catalogs, provider responses,
   credentials, or generated work directories.

Changes to the c-SKL kernel require numerical-regression evidence. Changes to
calibration, overlap, annotations, graph policy, or evidence presentation must
also update the relevant contract or method document.

Bug reports should include the smallest reproducible input contract, software
versions, command, structured error, and artifact or release identifier. Remove
credentials and personal or unpublished data before sharing logs.
