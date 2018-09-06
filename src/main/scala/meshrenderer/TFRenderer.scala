package meshrenderer

import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.{tf, _}

/**
  * Created by andreas on 8/15/18.
  */

case class TFRenderer(mesh: TFMesh, pts: Output, colors: Output, pose: TFPose, camera: TFCamera, illumination: Output, width: Int, height: Int) {

  val normals = TFMeshOperations.vertexNormals(pts.transpose(), mesh.triangles, mesh.trianglesForPointData)
  val worldNormals = {
    Transformations.poseRotationTransform(
      normals.transpose(), pose.pitch, pose.yaw, pose.roll
    ).transpose()
  }

  val ndcPts = Transformations.objectToNDC(pts, pose, camera)
  val ndcPtsTf = Transformations.ndcToTFNdc(ndcPts, width, height).transpose()

  val triangleIdsAndBCC = Rasterizer.rasterize_triangles(ndcPtsTf, mesh.triangles, width, height)
  val vtxIdxPerPixel = tf.gather(mesh.triangles, tf.reshape(triangleIdsAndBCC.triangleIds, Shape(-1)))

  val vtxIdxPerPixelGath = tf.gatherND(mesh.triangles, tf.expandDims(triangleIdsAndBCC.triangleIds, 2))

  val interpolatedAlbedo = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, colors, ndcPtsTf(::, 2))
  val interpolatedNormals = Shading.interpolateVertexDataPerspectiveCorrect(triangleIdsAndBCC, vtxIdxPerPixel, worldNormals, ndcPtsTf(::, 2))
  //val lambert = Renderer.lambertShader(interpolatedAlbedo, Tensor(0.5f, 0.5f, 0.5f), Tensor(0.5f, 0.5f, 0.5f), Tensor(Seq(0f,0f,1f)), interpolatedNormals)
  val shShader = Shading.sphericalHarmonicsLambertShader(interpolatedAlbedo, interpolatedNormals, illumination)
}

