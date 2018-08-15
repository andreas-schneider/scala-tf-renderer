package meshrenderer

import org.platanios.tensorflow.api.{Shape, Tensor}
import scalismo.faces.color.RGBA
import scalismo.faces.mesh.{ColorNormalMesh3D, VertexColorMesh3D}
import scalismo.mesh.{SurfacePointProperty, TriangleList}

/**
  * Created by andreas on 8/15/18.
  */

case class TFMesh(mesh: VertexColorMesh3D) {
  val trianglesForPointData = {
    val triForPoint = mesh.shape.pointSet.pointIds.toIndexedSeq.map { id =>
      (id, mesh.shape.triangulation.adjacentTrianglesForPoint(id))
    }
    TFMeshOperations.trianglesForPoint(triForPoint)
  }
  val adjacentPoints = TFMeshOperations.adjacentPoints(mesh.shape)
  val triangles = TFMesh.triangulationAsTensor(mesh.shape.triangulation)
  val pts = TFConversions.pointsToTensor(mesh.shape.position.pointData)//.transpose()
  val colors = TFConversions.pointsToTensor(mesh.color.pointData.map(_.toRGB)).transpose()
  val normals = TFConversions.pointsToTensor(mesh.shape.vertexNormals.pointData).transpose()
}

object TFMesh {
  def apply(mesh: ColorNormalMesh3D): TFMesh = {
    val vertexColor: SurfacePointProperty[RGBA] = SurfacePointProperty.averagedPointProperty(mesh.color)
    TFMesh(VertexColorMesh3D(mesh.shape, vertexColor))
  }

  def triangulationAsTensor(triangulation: TriangleList): Tensor = {
    val triangles = triangulation.triangles
    val data = Array.fill(triangles.length*3)(-1)
    for(i <- 0 until triangles.length) {
      val triangle = triangles(i)
      data(i*3) = triangle.ptId1.id
      data(i*3+1) = triangle.ptId2.id
      data(i*3+2) = triangle.ptId3.id
    }
    Tensor(data).reshape(Shape(triangles.length, 3))
  }
}