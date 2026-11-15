# Schema 0.1.0

The TypeScript transport contract follows section 4 of the P0a plan. `parseAnalysis` and `parseVideoAsset` throw readable errors for malformed input; `validate*` returns a boolean. Unknown additive fields are tolerated. An unsupported `schemaVersion` is rejected rather than interpreted silently. Additive fields require a minor schema release; breaking layouts require a major release with an explicit migration. `pipelineVersion` identifies algorithm changes; each metric carries a separate `definitionVersion`.

`validateSkeletonData` checks binary lengths, finite increasing capture timestamps, normalized coordinates, and confidence ranges. Float32 files use little-endian encoding. Missing coordinates are NaN with confidence 0. JSON never carries NaN values. `CWB_65` provides the index mapping with the face landmarks removed.

Production results should persist immutable original input, model/version metadata and binaries. Re-analysis creates a new analysis ID rather than overwriting old results. Worker/gateway transport errors must become an explicit failed task.
