# Notebook migration

Source: `03_PIM_efficient_net_v2_architecture_model_breakhis_classification.ipynb`, provided by the project author. The original file is left unchanged in its source location. Its outputs and absolute local paths are not copied into this repository.

| Notebook behavior | Repository behavior |
| --- | --- |
| Absolute dataset and checkpoint paths | JSON configuration |
| Shuffled batches split with take/skip | Stratified patient splits before batching |
| Random validation and test crops | Deterministic center crops during inference |
| ImageDataGenerator through numpy_function | Serializable Keras augmentation layers; shear omitted |
| Parent backbone remains non-trainable during fine-tuning | Parent enabled before selecting trainable child blocks |
| Latest directory entry treated as best model | Explicit best.keras checkpoint per stage |
| Test evaluation after every training stage | Validation selection and separate final test evaluation |
| Repeated inline orchestration per magnification | Configurable CLI with independent run directories |
| Full-model files loaded as weights | Full-model serialization and loading |

The shallow EfficientNetV2B1, Inception branches, flatten head, dropout, image dimensions, batch size, warm-up schedule, and staged block selection follow the notebook. Validation selection now uses loss; backbone normalization statistics remain fixed. Fine-tuning schedules restart their epoch count per stage. These changes require new experiments and invalidate direct comparison with previously reported notebook metrics.

The patient parser expects native BreaKHis filenames and uses the year/sample token as the grouping identity. It rejects unknown PNG filenames. The same seed and patient population produce matching splits across magnifications; verify patient membership before comparing runs with different populations. This implementation does not check image-content duplicates or provide cross-validation, confidence intervals, or automatic resume.
