
ThisBuild / version := "1.0"
ThisBuild / scalaVersion := "2.13.14"
ThisBuild / organization := "TinyML_NPU"

val spinalVersion = "1.12.3"
val spinalCore = "com.github.spinalhdl" %% "spinalhdl-core" % spinalVersion
val spinalLib = "com.github.spinalhdl" %% "spinalhdl-lib" % spinalVersion
val spinalIdslPlugin = compilerPlugin("com.github.spinalhdl" %% "spinalhdl-idsl-plugin" % spinalVersion)
lazy val projectname = (project in file("."))
  .settings(
    name := "tinyml-npu-hw",
    // Use standard SBT layout (the repo sources live under src/main/scala).
    // NOTE: previous path "hw/spinal" is not present in this repo.
    Compile / scalaSource := baseDirectory.value / "src" / "main" / "scala",
    libraryDependencies ++= Seq(
      spinalCore,
      spinalLib,
      spinalIdslPlugin
    )
  )


scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-language:postfixOps"
)
fork := true
