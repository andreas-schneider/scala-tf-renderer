package meshrenderer

import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.{tf, _}

/**
  * Created by andreas on 8/18/18.
  */

trait TFRenderParameterModel {
  val pts : Output
  val colors: Output
  val illumination: Output
  val pose: TFPose
  val camera: TFCamera
}

/* Models all variables as initialization + offset. Initialization is fixed and offset is variable. */
class OffsetFromInitializationModel(initPts: Tensor, initColors: Tensor, initPose: TFPoseTensor, initCamera: TFCameraTensor, initLight: Tensor) extends TFRenderParameterModel {

  //placeholders and variables
  val initalPoints = tf.placeholder(FLOAT32, initPts.shape, "initPts")
  val initialColors = tf.placeholder(FLOAT32, initColors.shape, "initColors")
  val initialIllumination = tf.placeholder(FLOAT32, Shape(9, 3), "illumination")
  val initialPoseRotation = tf.placeholder(FLOAT32, initPose.rotation.shape, "poseRotation")
  val initialPoseTranslation = tf.placeholder(FLOAT32, initPose.translation.shape, "poseTranslation")
  val initialCamera = tf.placeholder(FLOAT32, initCamera.parameters.shape, "camera")

  lazy val ptsVar = tf.variable("pointsOffset", FLOAT32, initPts.shape, tf.ZerosInitializer)
  lazy val colorsVar = tf.variable("colorsOffset", FLOAT32, initColors.shape, tf.ZerosInitializer)
  lazy val illumVar = tf.variable("illuminationOffset", FLOAT32, initialIllumination.shape, tf.ZerosInitializer)
  lazy val poseRotVar = tf.variable("poseRotationOffset", FLOAT32, initialPoseRotation.shape, tf.ZerosInitializer)
  lazy val poseTransVar = tf.variable("poseTranslationOffset", FLOAT32, initialPoseTranslation.shape, tf.ZerosInitializer)
  lazy val cameraVar = tf.variable("cameraOffset", FLOAT32, initialCamera.shape, tf.ZerosInitializer)

  lazy val illumOffset: Output = tf.identity(illumVar)
  lazy val colorsOffset: Output = tf.identity(colorsVar)
  lazy val ptsOffset: Output = tf.identity(ptsVar)
  lazy val poseRotationOffset: Output = tf.identity(poseRotVar)
  lazy val poseTranslationOffset: Output = tf.identity(poseTransVar)
  lazy val cameraOffset: Output = tf.identity(cameraVar)

  //val colors = tf.variable("color", FLOAT32, tfMesh.colors.shape, tf.RandomUniformInitializer())
  //val colors = tfMesh.colors

  lazy val pts = initalPoints + ptsOffset
  lazy val colors = initialColors + colorsOffset
  lazy val illumination = initialIllumination + illumOffset
  lazy val poseRotation = initialPoseRotation + poseRotationOffset
  lazy val poseTranslation = initialPoseTranslation + poseTranslationOffset
  lazy val pose = TFPose(poseRotation, poseTranslation)
  lazy val camera = TFCamera(initialCamera + cameraOffset)

  val feeds: FeedMap = Map(
    initalPoints -> initPts,
    initialColors -> initColors,
    initialIllumination -> initLight,
    initialPoseRotation -> initPose.rotation, initialPoseTranslation -> initPose.translation,
    initialCamera -> initCamera.parameters
  )
}