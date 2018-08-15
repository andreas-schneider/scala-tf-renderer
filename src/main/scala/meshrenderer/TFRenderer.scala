package meshrenderer

import org.platanios.tensorflow.api.{tf, _}

/**
  * Created by andreas on 8/15/18.
  */

case class TFRenderer(mesh: TFMesh, initPose: TFPoseTensor, initCamera: TFCameraTensor, initLight: Tensor, width: Int, height: Int) {
  val initPts = mesh.pts
  val initColors = mesh.colors

  //placeholders and variables
  val initalPoints = tf.placeholder(FLOAT32, initPts.shape, "initPts")
  val initialColors = tf.placeholder(FLOAT32, initColors.shape, "initColors")
  val initialIllumination = tf.placeholder(FLOAT32, Shape(9, 3), "illumination")
  val initialPoseRotation = tf.placeholder(FLOAT32, initPose.rotation.shape, "poseRotation")
  val initialPoseTranslation = tf.placeholder(FLOAT32, initPose.translation.shape, "poseTranslation")
  val initialCamera = tf.placeholder(FLOAT32, initCamera.parameters.shape, "camera")

  val ptsOffset = tf.variable("pointsOffset", FLOAT32, initPts.shape, tf.ZerosInitializer)
  val colorsOffset = tf.variable("colorsOffset", FLOAT32, initColors.shape, tf.ZerosInitializer)
  val illumOffset = tf.variable("illuminationOffset", FLOAT32, initialIllumination.shape, tf.ZerosInitializer)
  val poseRotationOffset = tf.variable("poseRotationOffset", FLOAT32, initialPoseRotation.shape, tf.ZerosInitializer)
  val poseTranslationOffset = tf.variable("poseTranslationOffset", FLOAT32, initialPoseTranslation.shape, tf.ZerosInitializer)
  val cameraOffset = tf.variable("cameraOffset", FLOAT32, initialCamera.shape, tf.ZerosInitializer)

  //val colors = tf.variable("color", FLOAT32, tfMesh.colors.shape, tf.RandomUniformInitializer())
  //val colors = tfMesh.colors

  val pts = initalPoints + ptsOffset
  val colors = initialColors + colorsOffset
  val illumination = initialIllumination + illumOffset
  val poseRotation = initialPoseRotation + poseRotationOffset
  val poseTranslation = initialPoseTranslation + poseTranslationOffset
  val pose = TFPose(poseRotation, poseTranslation)
  val camera = TFCamera(initialCamera + cameraOffset)

  val normals = TFMeshOperations.vertexNormals(pts.transpose(), mesh.triangles, mesh.trianglesForPointData)
  val worldNormals = {
    Transformations.poseRotationTransform(
      normals.transpose(), pose.pitch, pose.yaw, pose.roll
    ).transpose()
  }

  val ndcPts = Transformations.objectToNDC(pts, pose, camera)
  val ndcPtsTf = Transformations.ndcToTFNdc(ndcPts, width, height).transpose()

  val triangleIdsAndBCC = Rasterizer.rasterize_triangles(ndcPtsTf, mesh.triangles, width, height)
  val vtxIdxPerPixel = tf.gather(mesh.triangles, tf.reshape(triangleIdsAndBCC.triangleIds, Shape(-1)), name = "226")

  val vtxIdxPerPixelGath = tf.gatherND(mesh.triangles, tf.expandDims(triangleIdsAndBCC.triangleIds, 2))

  val interpolatedAlbedo = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, colors, ndcPtsTf(::, 2))
  val interpolatedNormals = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, worldNormals, ndcPtsTf(::, 2))
  //val lambert = Renderer.lambertShader(interpolatedAlbedo, Tensor(0.5f, 0.5f, 0.5f), Tensor(0.5f, 0.5f, 0.5f), Tensor(Seq(0f,0f,1f)), interpolatedNormals)
  val shShader = Shading.sphericalHarmonicsLambertShader(interpolatedAlbedo, interpolatedNormals, illumination)

  val feeds = Map(
    initalPoints -> initPts,
    initialColors -> initColors,
    initialIllumination -> initLight,
    initialPoseRotation -> initPose.rotation, initialPoseTranslation -> initPose.translation,
    initialCamera -> initCamera.parameters
  )
}

