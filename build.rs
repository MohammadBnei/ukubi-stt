// tonic 0.14 split prost codegen out of tonic-build into tonic-prost-build,
// and the generated code references `tonic_prost::ProstCodec` — so both the
// build-dependency and the runtime `tonic-prost` dependency are required.
// A missing runtime one fails at compile of the generated module, not here.
//
// The descriptor set is what gRPC reflection serves: it lets a client discover
// the service and its messages without being handed a .proto, which is the
// difference between "vendor this file first" and `grpcurl <host> list`.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    tonic_prost_build::configure()
        .file_descriptor_set_path(out.join("stt_descriptor.bin"))
        .compile_protos(&["proto/stt/v1/stt.proto"], &["proto"])?;
    Ok(())
}
