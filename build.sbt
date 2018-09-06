organization  := "ch.unibas.cs.gravis"
name := """scala-tf-renderer"""

scalaVersion  := "2.12.2"
scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

resolvers += Resolver.bintrayRepo("unibas-gravis", "maven")

resolvers +="Statismo (private)" at "https://statismo.cs.unibas.ch/repository/private/"

resolvers += Opts.resolver.sonatypeSnapshots

credentials +=Credentials(Path.userHome /".ivy2" /".credentials-statismo-private")

libraryDependencies  ++= Seq(
  "org.platanios" %% "tensorflow" % "0.2.0",
    "org.platanios" %% "tensorflow" % "0.2.0" classifier "linux-gpu-x86_64",
  //"org.platanios" %% "tensorflow" % "0.2.4" classifier "linux-cpu-x86_64",

//  "org.bytedeco.javacpp-presets" % "opencv-platform" % "3.4.1-1.4.1",
//  "org.bytedeco.javacpp-presets" % "opencv" % "3.4.1-1.4.1" classifier "linux-x86_64",
//  "org.platanios" %% "tensorflow-data" % "0.1.1"

//    "ch.unibas.cs.gravis" %% "faces-scala" % "0.9.0-RC1", 
//    "ch.unibas.cs.gravis" %% "faces-scala-prt" % "develop-e275b07b5012ae42f3ddf59c155c1f16a923cfd0-SNAPSHOT", //"develop-ea54f8270d98985266b667e810b18aedd1383bf5-SNAPSHOT",
//    "ch.unibas.cs.gravis" %% "scalismo" % "0.15.0",
//    "ch.unibas.cs.gravis" % "scalismo-native-all" % "3.0.0",
      "ch.unibas.cs.gravis" %% "scalismo-faces" % "0.9.1",
//    "ch.unibas.cs.gravis" %% "basel-face-pipeline" % "0.1-SNAPSHOT",
//    "ch.unibas.cs.gravis" %% "faces-utils" % "develop-9c9dd84ab7ed3eeb57c65bb9b9d434d7c059921c",//"develop-8efeaf67947a8a964084064b6f47ff96605665d5",
//"ch.unibas.cs.gravis" %% "scalismo-ui" % "0.11.+",
    "org.scalatest" %% "scalatest" % "3.0.0" % "test"
)

//libraryDependencies += "org.sameersingh.scalaplot" % "scalaplot" % "0.0.4"

// git versioning with sbt-git plugin
enablePlugins(GitVersioning, GitBranchPrompt)
git.baseVersion := "develop"
com.typesafe.sbt.SbtGit.useJGit

// assembly 
import AssemblyKeys._
assemblySettings
mainClass in assembly := None

mergeStrategy in assembly := {
  case PathList("org", "spire-math", "spire", xs @ _*) => MergeStrategy.first
  case x => (mergeStrategy in assembly).value(x)
}
