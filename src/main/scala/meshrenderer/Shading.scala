package meshrenderer

import meshrenderer.Rasterizer.RasterizationOutput
import org.platanios.tensorflow.api.{tf, _}
import scalismo.faces.numerics.SphericalHarmonics
import scalismo.faces.parameters.SphericalHarmonicsLight


/**
  * Created by andreas on 8/8/18.
  */
object Shading {

  /**
    * Interpolate vertex data [width x height x #attributes] with perspective correction i.e. interpolates in world space.
    * @param r
    * @param vtxIdxPerPixel [widht x height x number of vertices a pixel can belong to (triangle: 3)]
    * @param vertexData  [#vertices x #attributes]
    * @param zVertexData [width x height]
    * @return
    */
  def interpolateVertexDataPerspectiveCorrect(r: RasterizationOutput, vtxIdxPerPixel: Output, vertexData: Output, zVertexData: Output): Output = {
    val attrDim = vertexData.shape(1)

    val zVertexDataExp = {
      tf.tile(tf.expandDims(zVertexData, 1), Seq(1, 3))
    }

    val vertexDataCorrected = vertexData / zVertexDataExp //divide by the dpeth of the vertex for correcting perspective

    val valuesPerPixel = tf.transpose(tf.gather(vertexDataCorrected, vtxIdxPerPixel, name = "gatherInInterpolateVertexData"), Seq(0,2,1))

    /*println("vertexData", vertexData)
    println("vtxIdxPerPixel", vtxIdxPerPixel.shape)
    println("valuesPerPixel", valuesPerPixel)*/

    val barycentricCoordinates = r.barycetricImage.reshape(Shape(-1,3))
    val baryRepeated = tf.tile(
      barycentricCoordinates,
      Seq(1,attrDim) // repeat barycentric coordinates x attributes
    ).reshape(Shape(-1, attrDim, 3))

    val interpolateValues = tf.sum(baryRepeated * valuesPerPixel, 2)
    val expanded  = tf.expandDims(r.zBufferImage, 2)
    val depth = tf.tile(
      expanded,
      Seq(1,1,attrDim)
    )

    tf.reshape(interpolateValues, Shape(r.triangleIds.shape(0), r.triangleIds.shape(1), attrDim)) * depth // multiply by per pixel depth value, perspective is now corrected.
  }

  /** Use this only to interpolate vertex data [width x height x attributes] with screen space barycentric coordinates.
    *
    * @param r
    * @param screenSpaceBarycentricCoordinates [width x height x 3]
    * @param vtxIdxPerPixel [width x height x #number of vertices a pixel can belong to (triangle: 3)]
    * @param vertexData [#vertex x #attributes].
    * @return
    */
  def interpolateVertexDataWithoutPerspectiveCorrection(r: RasterizationOutput, screenSpaceBarycentricCoordinates: Output, vtxIdxPerPixel: Output, vertexData: Output): Output = {

    val valuesPerPixel = tf.transpose(tf.gather(vertexData, vtxIdxPerPixel, name = "gatherInInterpolateVertexData"), Seq(0,2,1))

    println("vertexData", vertexData)
    println("vtxIdxPerPixel", vtxIdxPerPixel.shape)
    println("valuesPerPixel", valuesPerPixel)

    val barycentricCoordinates = screenSpaceBarycentricCoordinates.reshape(Shape(-1,3))
    val baryRepeated = tf.tile(
      barycentricCoordinates,
      Seq(1,vertexData.shape(1)) // repeat barycentric coordinates x attributes
    ).reshape(Shape(-1, vertexData.shape(1), 3))

    println("baryRepeated", baryRepeated)

    val interpolateValues = tf.sum(baryRepeated * valuesPerPixel, 2)

    println("interpolateValues", interpolateValues)

    tf.reshape(interpolateValues, Shape(r.triangleIds.shape(0), r.triangleIds.shape(1), vertexData.shape(1)))
  }

  def lambertShader(albedoPerPixel: Output, ambientLight: Output, diffuseLight: Output, lightDir: Output,normalsPerPixel: Output) = {
    val normals = normalsPerPixel.reshape(Shape(-1, 3))
    val zero = tf.zeros(normals.dataType, Shape(normals.shape(0), 1))
    val dot = tf.matmul(normals, lightDir.transpose())
    val normalsWithZero = tf.concatenate(Seq(dot, zero), axis=1)
    val diffuseFactor = tf.max(normalsWithZero, Seq(1), true)
    val diffuse = diffuseFactor * diffuseLight
    tf.add(diffuse, ambientLight).reshape(Shape(albedoPerPixel.shape(0), albedoPerPixel.shape(1), 3)) * albedoPerPixel
  }

  def shBasis(dir: Output) = {
    val N0 : Float = SphericalHarmonics.N0.toFloat
    val N1 : Float = SphericalHarmonics.N1.toFloat
    val N2_1 : Float = SphericalHarmonics.N2_1.toFloat
    val N2_0 : Float = SphericalHarmonics.N2_0.toFloat
    val N2_2 : Float = SphericalHarmonics.N2_2.toFloat
    val x = dir(::, 0)
    val y = dir(::,1)
    val z = dir(::,2)
    val one = tf.onesLike(x)
    val xx = tf.multiply(x,x)
    val yy = tf.multiply(y,y)
    val zz = tf.multiply(z,z)

    val sh = tf.stack(Seq(
      N0*one,

      N1*y, N1*z, N1*x,

      N2_1*x*y,
      N2_1*y*z,
      N2_0*(2f*zz-xx-yy),
      N2_1*z*x,
      N2_2*(xx-yy)
    ))

    println("sh", sh)

    sh
  }

  def sphericalHarmonicsLambertShader(albedoPerPixel: Output, normalsPerPixel: Output, envMap: Output) = {
    val nCoeffs = envMap.shape(0)
    val sh = shBasis(normalsPerPixel.reshape(Shape(-1,3)))
    val lk = Tensor(SphericalHarmonicsLight.lambertKernel.map(_.toFloat)).reshape(Shape(9,1))
    val shlk = sh*lk
    println("shlk", shlk)
    println("envMap", envMap)

    val envRep = tf.tile(tf.expandDims(envMap, axis=2), Seq(1,1, shlk.shape(1))).transpose(Seq(0,2,1))
    println("envRep", envRep)

    val expShlk = tf.tile(tf.expandDims(shlk, axis=2), Shape(1,1,3))
    println("expShlk", expShlk)

    val res = expShlk * envRep
    println("res", res)

    val convolved = tf.sum(res, axes=Seq(0))
    println("convolved", convolved)

    val conv = convolved.reshape(Shape(albedoPerPixel.shape(0), albedoPerPixel.shape(1), 3))
    println("conv", conv)
    println("albedoPerPixel", albedoPerPixel)

    tf.createWithNameScope("shshader") {
      albedoPerPixel * conv
    }
  }

}