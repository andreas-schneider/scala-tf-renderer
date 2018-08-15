package meshrenderer

import java.io.File

import org.platanios.tensorflow.api.ops.Gradients
import org.platanios.tensorflow.api.ops.Gradients.Registry
import org.platanios.tensorflow.api.{tf, _}
import scalismo.faces.color.{RGB, RGBA}
import scalismo.faces.io.{MeshIO, MoMoIO, PixelImageIO, RenderParameterIO}
import scalismo.faces.mesh.{ColorNormalMesh3D, VertexColorMesh3D}
import scalismo.faces.parameters._
import scalismo.geometry.Point
import scalismo.mesh.TriangleMesh3D


/**
  * Created by andreas on 8/8/18.
  */
object Example {
  def main(args: Array[String]): Unit = {
    Registry.register("RasterizeTriangles", Rasterizer.rasterizeTrianglesGrad)

    /*val param = RenderParameter.defaultSquare.withPose(Pose(1.0, Vector(0,0,-1000), 0,1.1,0)).fitToImageSize(200,250).withEnvironmentMap(
      SphericalHarmonicsLight(SphericalHarmonicsLight.frontal.coefficients ++ IndexedSeq(Vector(0.5,0.5,0.1), Vector(0.2,0.1,0.7), Vector(0.0,0.2,0.0), Vector(0.2,0.1,-0.1), Vector(-0.1,-0.1,-0.1)))
    )*/
    //
    val imageFn = new File("/home/andreas/work/Code/phd-experiments/targets/collection/jimmy_carter/JimmyCarterPortrait2.png")
    val image = {
      val img = PixelImageIO.read[RGB](imageFn).get
      img.resample((img.width*0.1).toInt, (img.height*0.1).toInt)
    }
    val param = RenderParameterIO.read(new File("../phd-experiments/out/fits/jimmy_carter/JimmyCarterPortrait2/fit-best.rps")).get.fitToImageSize(image.width, image.height)
    //val image = PixelImageIO.read[RGB](new File("/tmp/tf_rendering_sh_lambert.png")).get

    val model = {
      scalismo.initialize()
      val momoFn = new File("model2017-1_bfm_nomouth.h5")
      MoMoIO.read(momoFn).get
    }

    //val mesh = MeshIO.read(new File("/home/andreas/export/mean2012_l7_bfm_pascaltex.msh.gz")).get.colorNormalMesh3D.get
    val mesh = model.instance(param.momo.coefficients)

    MeshIO.write(mesh, new File("/tmp/jimmi_fit.ply")).get
    val tfMesh = TFMesh(mesh)

    val initPose = TFPose(param.pose)
    val initCamera = TFCamera(param.camera)
    val initLight = TFConversions.pointsToTensor(param.environmentMap.coefficients).transpose()

    val renderer = TFRenderer(tfMesh, initPose, initCamera, initLight, param.imageSize.width, param.imageSize.height)

    def renderInitialParametersAndCompareToGroundTruth() = {
      val sess = Session()
      sess.run(targets=tf.globalVariablesInitializer())

      val result = sess.run(
        feeds = renderer.feeds,
        fetches = Seq(renderer.shShader, renderer.worldNormals)
      )

      val tensorImg = result(0).toTensor
      val img = TFConversions.oneToOneTensorImage3dToPixelImage(tensorImg)
      PixelImageIO.write[RGB](img, new File("/tmp/fit.png")).get

      {
        val img = ParametricRenderer.renderParameterMesh(param, ColorNormalMesh3D(mesh), RGBA.Black)
        PixelImageIO.write[RGBA](img, new File("/tmp/fitgt.png")).get
      }
    }

    val targetImage = TFConversions.image3dToTensor(image)

    val test = TFConversions.oneToOneTensorImage3dToPixelImage(targetImage)
    PixelImageIO.write(test, new File("/tmp/test.png")).get

    renderInitialParametersAndCompareToGroundTruth()

    //loss
    val target = tf.placeholder(FLOAT32, Shape(param.imageSize.height, param.imageSize.width, 3), "target")
    val reg = TFMeshOperations.vertexToNeighbourDistance(tfMesh.adjacentPoints, renderer.normals)
    val rec = tf.sum(tf.mean(tf.abs(target - renderer.shShader)))
    println("reg", reg)
    println("rec", rec)
    val loss = rec //+ reg

    println("loss", loss)

    val grad: Seq[OutputLike] = Gradients.gradients(
      Seq(loss),
      Seq(
        //renderer.colorsOffset,
        renderer.ptsOffset
      )
    )
    val optimizer = tf.train.AMSGrad(0.1, name="adal")
    val optFn = optimizer.applyGradients(
      Seq(
      (grad(0), renderer.ptsOffset)
    ))

    val session = Session()
    session.run(targets=tf.globalVariablesInitializer())

    for(i <- 0 until 60) {
      val result = session.run(
        feeds = Map(target -> targetImage) ++ renderer.feeds,
        fetches = Seq(renderer.shShader, loss, reg, rec),
        targets = optFn
      )

      println(s"iter ${i}", result(1).toTensor.summarize(), result(2).toTensor.summarize(), result(3).toTensor.summarize())

      if(i % 30 == 0) {
        val rendering = TFConversions.oneToOneTensorImage3dToPixelImage(result(0).toTensor)
        PixelImageIO.write(rendering, new File(s"/tmp/${i}_tf_rendering.png")).get
      }
    }

    val fetch = session.run(
      feeds = Map(target -> targetImage) ++ renderer.feeds,
      fetches = Seq(renderer.pts, loss)
    )

    val finalMesh = {
      val vtx = {
        val finalPts = fetch(0).toTensor
        val n = finalPts.shape(1)

        for (i <- 0 until n) yield {
          val x = finalPts(0, i).entriesIterator.toIndexedSeq(0).asInstanceOf[Float].toDouble
          val y = finalPts(1, i).entriesIterator.toIndexedSeq(0).asInstanceOf[Float].toDouble
          val z = finalPts(2, i).entriesIterator.toIndexedSeq(0).asInstanceOf[Float].toDouble
          Point(x, y, z)
        }
      }

      TriangleMesh3D(vtx, mesh.shape.triangulation)
    }

    val finalFullMesh = VertexColorMesh3D(finalMesh, mesh.color)
    MeshIO.write(finalFullMesh, new File("/tmp/jimmi.ply")).get

  }
}
