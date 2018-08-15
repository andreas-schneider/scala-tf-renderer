package meshrenderer

import org.platanios.tensorflow.api.{tf, _}
import scalismo.faces.parameters.{ImageSize, Pose, RenderParameter}
import scalismo.geometry.{Point, Vector, _3D}
import scalismo.mesh.TriangleMesh3D
import scalismo.utils.Random


object Transformations {

  def poseTransform(pt: OutputLike, pitch: OutputLike, yaw: OutputLike, roll: OutputLike, translation: OutputLike) = {
    poseRotationTransform(pt, pitch, yaw, roll) + translation
  }

  def poseRotationTransform(pt: OutputLike, pitch: OutputLike, yaw: OutputLike, roll: OutputLike) = {
    val X = {
      val pitchc = tf.cos(pitch)
      val pitchs = tf.sin(pitch)
      tf.stack(Seq(1f, 0f, 0f,
        0f, pitchc, -pitchs,
        0f, pitchs, pitchc
      )).reshape(Shape(3,3))
    }

    val Y = {
      val yawc = tf.cos(yaw)
      val yaws = tf.sin(yaw)
      tf.stack(Seq(yawc, 0f, yaws,
        0f, 1f, 0f,
        -yaws, 0f, yawc)).reshape(Shape(3,3))
    }

    val Z = {
      val rollc = tf.cos(roll)
      val rolls = tf.sin(roll)
      tf.stack(Seq(rollc, -rolls, 0f,
        rolls, rollc, 0f,
        0f, 0f, 1f
      )).reshape(Shape(3,3))
    }
    tf.matmul(Z, tf.matmul(Y,tf.matmul(X,pt)))
  }

  def projectiveTransformation(pt: OutputLike,
                               near: OutputLike, far: OutputLike,
                               sensorSizeX: OutputLike, sensorSizeY: OutputLike,
                               focalLength: OutputLike,
                               principalPointX: OutputLike, principalPointY: OutputLike) = {
    val f = far
    val n = near
    val ppx = principalPointX
    val ppy = principalPointY
    val fl = focalLength
    val ssx = sensorSizeX
    val ssy = sensorSizeY
    val px = pt(0, ::)
    val py = pt(1, ::)
    val pz = pt(2, ::)

    val newpx = ppx -  (px * 2f * fl)/(pz * ssx)
    val newpy = ppy -  (py * 2f * fl)/(pz * ssy)
    val newpz = (f * n * 2f / pz + n + f)/(f-n)
    tf.stack(Seq(newpx, newpy, newpz), axis=1).transpose()//.reshape(Shape(3,2))
  }

  def screenTransformation(pt: OutputLike, width: OutputLike, height: OutputLike) = {
    val n = 0f
    val f = 1f
    val px = pt(0, ::)
    val py = pt(1, ::)
    val pz = pt(2, ::)
    val newpx = (px + 1f) * width / 2f
    val newpy = (-py + 1f) * height / 2f
    val newpz = pz * (f-n)/2f+(f+n)/2f
    tf.stack(Seq(newpx, newpy, newpz), axis=1).transpose()
  }

  def objectToNDC(pts: OutputLike, pose: TFPose, camera: TFCamera) = {
    val poseTransformed = poseTransform(pts, pose.pitch, pose.yaw, pose.roll, pose.translation)

    projectiveTransformation(
      poseTransformed,
      camera.near, camera.far,
      camera.sensorSizeX, camera.sensorSizeY,
      camera.focalLength,
      camera.principalPointX, camera.principalPointY
    )
  }

  /** normalized device coordinates of rasterizer are different than in the scalismo-faces renderer.*/
  def ndcToTFNdc(pts: OutputLike, width: Int, height: Int) = {
    val px = pts(0, ::)
    val py = pts(1, ::)
    val pz = pts(2, ::)
    val yoffset = 1f/height//there is a 0.5 pixel diagonal shift between the tf mesh rasterizer and the scalismo-faces renderer,
    val xoffset = 1f/width
    //tf.stack(Seq(-py+offset, px, pz), axis=1).transpose()
    tf.stack(Seq(px+xoffset, -py+yoffset, pz))
  }

  def renderTransform(pts: OutputLike, pose: TFPose, camera: TFCamera, width: Output, height: Output) = {
    val toNDC = objectToNDC(pts, pose, camera)
    screenTransformation(toNDC, width, height)
  }

  def bccScreenToWorldCorrection(bccScreen: OutputLike, zBuffer: OutputLike, zCoordinatesPerVertex: OutputLike) = {
    val d = zBuffer
    val z1 = zCoordinatesPerVertex(::,::,0)
    val z2 = zCoordinatesPerVertex(::,::,1)
    val z3 = zCoordinatesPerVertex(::,::,2)
    val a = bccScreen(::,::,0)
    val b = bccScreen(::,::,1)
    val c = bccScreen(::,::,2)
    //val d = z2*z3 + z3*b * (z1-z2) + z2*c * (z1-z3)
    val dIsZero = tf.where(tf.equal(d, 0f))
    val newBccDIsZero = bccScreen * dIsZero
    val newB = z1*z3*b / d
    val newC = z1*z2*c / d
    val newBCC = tf.stack(Seq(1-newB-newC, newB, newC))
    newBCC * (1f-dIsZero) + newBccDIsZero
  }

  def testScreenTransform(pts: Tensor, ptsScal: IndexedSeq[Point[_3D]]) = {
    val camera = RenderParameter.defaultSquare.camera
    val session = Session()
    //val pts = tf.stack(Seq(Seq(tfPt(0), tfPt(1), tfPt(2)), Seq(tfPt2(0), tfPt2(1), tfPt2(2))), axis=1).reshape(Shape(3,2))
    val w = RenderParameter.default.imageSize.width
    val h = RenderParameter.default.imageSize.height
    val screenTrf = screenTransformation(pts, w, h)
    val ret = session.run(fetches = Seq(screenTrf))
    val tfRes = ret.toTensor
    println(tfRes.summarize())
    val gt = RenderParameter.default.imageSize.screenTransform(ptsScal(0))
    println("point:", ptsScal(0))
    println("ground truth: ", gt)
    println("result: ", tfRes)
    require(tfRes(0,0,0).entriesIterator.toIndexedSeq(0).isInstanceOf[Float])
    require(
      withinEps(gt.x, tfRes(0, 0, 0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]) &&
      withinEps(gt.y, tfRes(0, 1, 0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]) &&
      withinEps(gt.z, tfRes(0, 2, 0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float])
    )
  }

  def testPerspective(pts: Tensor, ptsScal: IndexedSeq[Point[_3D]]) = {
    val camera = RenderParameter.defaultSquare.camera
    val session = Session()
    val proj = projectiveTransformation(pts, camera.near.toFloat, camera.far.toFloat, camera.sensorSize.x.toFloat, camera.sensorSize.y.toFloat, camera.focalLength.toFloat, camera.principalPoint.x.toFloat, camera.principalPoint.y.toFloat)
    val ret = session.run(fetches = Seq(proj))
    val tfRes = ret.toTensor
    val gt = camera.projection(ptsScal(0))
    println("point:", ptsScal(0))
    println("ground truth: ", gt)
    println("result: ", tfRes.summarize())
    println(pts.dataType)
    require(tfRes(0,0,0).entriesIterator.toIndexedSeq(0).isInstanceOf[Float])
    require(
      withinEps(gt.x, tfRes(0,0,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]) &&
      withinEps(gt.y, tfRes(0,1,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]) &&
      withinEps(gt.z, tfRes(0,2,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float])
    )
  }

  def withinEps(a: Double,b: Float): Boolean = {
    math.abs(a-b) < 0.0001f
  }

  def testPose(pts: Tensor, ptsScal: IndexedSeq[Point[_3D]]) = {
    val yaw = 1f
    val pitch = 0.1f
    val roll = -0.1f
    val translation = Vector(-10,100,-1000)
    val pose = Pose(1.0, translation, roll, yaw, pitch)

    val session = Session()
    val poseTrf = poseTransform(pts, pitch, yaw, roll, TFConversions.vec2Output(translation))

    val ret = session.run(fetches = Seq(poseTrf))
    val tfRes = ret.toTensor

    val gt = pose.transform(ptsScal(0))
    println("point:", ptsScal(0))
    println("ground truth: ", gt)
    println("result: ", tfRes)
    println(tfRes.summarize())
    require(tfRes(0,0,0).entriesIterator.toIndexedSeq(0).isInstanceOf[Float])
    require(
      withinEps(gt.x, tfRes(0,0,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float] ) &&
        withinEps(gt.y, tfRes(0,1,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]) &&
        withinEps(gt.z, tfRes(0,2,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float] )
    )
  }

  def testFull(pts: Tensor, ptsScal: IndexedSeq[Point[_3D]]) = {
    val session = Session()

    val rnd = Random(100000L)

    val tx = 0 + rnd.scalaRandom.nextDouble()*20
    val ty = 0 + rnd.scalaRandom.nextDouble()*20
    val tz = - 1000 + rnd.scalaRandom.nextDouble()*20
    val roll = rnd.scalaRandom.nextDouble()
    val yaw = rnd.scalaRandom.nextDouble()
    val pitch = rnd.scalaRandom.nextDouble()
    val w = rnd.scalaRandom.nextInt(50) + 50
    val h = rnd.scalaRandom.nextInt(50) + 50
    val pose = Pose(1.0, Vector(tx,ty,tz), roll, yaw, pitch)

    val param = RenderParameter.defaultSquare.withPose(pose).withImageSize(ImageSize(w,h))

    val inScreen = renderTransform(pts, TFPose(TFPose(param.pose)), TFCamera(TFCamera(param.camera)), w, h)

    val result = session.run(fetches = Seq(inScreen)).toTensor

    val gt = param.renderTransform(ptsScal(0))
    require(result(0,0,0).entriesIterator.toIndexedSeq(0).isInstanceOf[Float])
    require(withinEps(gt.x, result(0,0,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]) &&
      withinEps(gt.y, result(0,1,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]) &&
      withinEps(gt.z, result(0,2,0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float])
    )
  }

  def testBCCScreenToWorld(mesh: TriangleMesh3D) = {
    /*val param = RenderParameter.defaultSquare
    val corr = TriangleRenderer.renderCorrespondenceImage(mesh, param.pointShader, param.imageSize.width, param.imageSize.height)
    val px = corr(256,256)
    px.get.worldBCC*/
  }

  def main(args: Array[String]): Unit = {
    val rnd = Random(100000L)
    def rndP() = Point(rnd.scalaRandom.nextDouble()*10,rnd.scalaRandom.nextDouble(),rnd.scalaRandom.nextDouble()*10)
    val pt = rndP()
    val pt2 = rndP()//Point(2,-1,5)
    val ptsScal = IndexedSeq(pt, pt2)
    val tfPt = TFConversions.pt2Output(pt)
    val tfPt2 = TFConversions.pt2Output(pt2)
    val pts = Tensor(tfPt, tfPt2).transpose().reshape(Shape(3,2))

    //val pts2 = tf.stack(Seq(Seq(tfPt(0), tfPt(1), tfPt(2)), Seq(tfPt2(0), tfPt2(1), tfPt2(2))), axis=1).reshape(Shape(3,2))
    println(pts.shape)
    //println(pts2.shape)

    testPose(pts, ptsScal)
    testPerspective(pts, ptsScal)
    testScreenTransform(pts, ptsScal)

    testFull(pts, ptsScal)
  }
}




