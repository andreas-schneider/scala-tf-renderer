package meshrenderer

import org.platanios.tensorflow.api.tensors.TensorLike
import org.platanios.tensorflow.api.{Shape, Tensor, tf, _}
import scalismo.common.Vectorizer
import scalismo.faces.color.RGB
import scalismo.faces.image.{ImageBuffer, PixelImage, PixelImageDomain}
import scalismo.faces.parameters.{Camera, Pose}
import scalismo.geometry.{Point, Vector, _3D}

/**
  * Created by andreas on 8/15/18.
  *
  * Convert between tensorflow and scalismo-faces data structures.
  */
object TFConversions {

  def pt2Output(pt: Point[_3D]) = {
    Tensor(pt.x.toFloat, pt.y.toFloat, pt.z.toFloat).reshape(Shape(3,1))
  }

  def vec2Output(pt: Vector[_3D]) = {
    tf.stack(Seq(pt.x.toFloat, pt.y.toFloat, pt.z.toFloat)).reshape(Shape(3,1))
  }

  def pt2Tensor(pt: Point[_3D]) = {
    Tensor(pt.x.toFloat, pt.y.toFloat, pt.z.toFloat).reshape(Shape(3,1))
  }

  def vec2Tensor(pt: Vector[_3D]) = {
    Tensor(Seq(pt.x.toFloat, pt.y.toFloat, pt.z.toFloat)).reshape(Shape(3,1))
  }

  def pointsToTensor[A](points: IndexedSeq[A])(implicit vectorizer: Vectorizer[A]): Tensor = {
    val d = vectorizer.dim
    val data = Array.fill(points.length*d)(0f)
    for(i <- 0 until points.length) {
      val vec = vectorizer.vectorize(points(i))
      for(j <- 0 until d) {
        data(i * d + j) = vec(j).toFloat
      }
    }
    Tensor(data).reshape(Shape(points.length, 3)).transpose()
  }

  def image3dToTensor(image: PixelImage[RGB]) = {
    val res = Array.fill(image.height, image.width ,3 )(0.0f)
    for(r <- 0 until image.height) {
      for(c <- 0 until image.width) {
        val px = image(c,r)
        res(r)(c)(0) = px.r.toFloat
        res(r)(c)(1) = px.g.toFloat
        res(r)(c)(2) = px.b.toFloat
      }
    }
    println(res)
    val ret = Tensor(res)
    println(ret.summarize())

    ret(0)
  }

  def tensorImage3dToPixelImage(dataRaw: TensorLike, domain: PixelImageDomain) = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.asInstanceOf[Float].toDouble
    }
    sequenceImage3dToPixelImage(data, domain)
  }

  def oneToOneTensorImage3dToPixelImage(dataRaw: Tensor) = {
    //require( == domain.width &&  == domain.height)
    val w = dataRaw.shape(1)
    val h = dataRaw.shape(0)
    val buffer = ImageBuffer.makeInitializedBuffer(w, h)(RGB.Black)
    var r =0
    while(r < h){
      var c=0
      while(c < w) {
        buffer(c, r) = RGB(
          dataRaw(r, c, 0).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]
          ,
          dataRaw(r, c, 1).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]
          ,
          dataRaw(r, c, 2).entriesIterator.toIndexedSeq(0).asInstanceOf[Float]
        )
        c+=1
      }
      r+=1
    }
    buffer.toImage
  }

  def tensorImage3dIntToPixelImage(dataRaw: TensorLike, domain: PixelImageDomain) = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.asInstanceOf[Int].toDouble
    }
    sequenceImage3dToPixelImage(data, domain)
  }

  def tensorIntImage3dToPixelImage(dataRaw: TensorLike, domain: PixelImageDomain) = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.asInstanceOf[Int].toDouble
    }
    sequenceImage3dToPixelImage(data, domain)
  }

  def sequenceImage3dToPixelImage(data: Seq[Double], domain: PixelImageDomain) = {
    require(data.size == domain.width*domain.height*3)
    val w = domain.width
    val h = domain.height
    PixelImage(domain.width, domain.height, (x, y) => {
      val first = domain.index(x, y) * 3
      val second = domain.index(x, y) * 3 + 1
      val third = domain.index(x, y) * 3 + 2
      RGB(data(first), data(second), data(third))
    })
  }

  def tensorImage1dToPixelImage(dataRaw: TensorLike, domain: PixelImageDomain) = {
    val data: Seq[Double] = dataRaw.entriesIterator.toIndexedSeq.map {
      _.asInstanceOf[Float].toDouble
    }
    sequenceImage1dToPixelImage(data, domain)
  }

  def sequenceImage1dToPixelImage(data: Seq[Double], domain: PixelImageDomain) = {
    require(data.size == domain.width*domain.height)
    PixelImage(domain.width, domain.height, (x, y) => {
      data(domain.index(x, y))
    })
  }
}

//row vectors
case class TFPose(rotation: Output, translation: Output) {
  val yaw = rotation(0,0)
  val pitch = rotation(0,1)
  val roll = rotation(0,2)
}

case class TFPoseTensor(rotation: Tensor, translation: Tensor) {
  val yaw = rotation(0, 0)
  val pitch = rotation(0, 1)
  val roll = rotation(0, 2)
}

object TFPose {
  def apply(pose: Pose) = TFPoseTensor(
    Tensor(Seq(pose.yaw.toFloat, pose.pitch.toFloat, pose.roll.toFloat)),
    TFConversions.vec2Tensor(pose.translation)
  )
  def apply(pose: TFPoseTensor): TFPose = TFPose(pose.rotation, pose.translation)
}

case class TFCamera(parameters: OutputLike) {
  val focalLength = parameters(0,0)
  val principalPointX = parameters(0,1)
  val principalPointY = parameters(0,2)
  val sensorSizeX = parameters(0,3)
  val sensorSizeY = parameters(0,4)
  val near = parameters(0,5)
  val far = parameters(0,6)
}

case class TFCameraTensor(parameters: Tensor) {
  val focalLength = parameters(0,0)
  val principalPointX = parameters(0,1)
  val principalPointY = parameters(0,2)
  val sensorSizeX = parameters(0,3)
  val sensorSizeY = parameters(0,4)
  val near = parameters(0,5)
  val far = parameters(0,6)
}

object TFCamera {
  def apply(cam: Camera): TFCameraTensor = TFCameraTensor(
    Tensor(Seq(
    cam.focalLength.toFloat,
    cam.principalPoint.x.toFloat, cam.principalPoint.y.toFloat,
    cam.sensorSize.x.toFloat, cam.sensorSize.y.toFloat,
    cam.near.toFloat,
    cam.far.toFloat
    )
    ) //row vector
  )
  def apply(cam: TFCameraTensor): TFCamera = TFCamera(cam.parameters)
}