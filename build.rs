// tonic 0.14 split prost codegen out of tonic-build into tonic-prost-build,
// and the generated code references `tonic_prost::ProstCodec` — so both the
// build-dependency and the runtime `tonic-prost` dependency are required.
// A missing runtime one fails at compile of the generated module, not here.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("proto/stt/v1/stt.proto")?;
    Ok(())
}
