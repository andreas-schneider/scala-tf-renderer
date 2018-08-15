package meshrenderer

import java.io.File

import org.platanios.tensorflow.api.{tf, _}
import scalismo.common.PointId
import scalismo.faces.io.MeshIO
import scalismo.mesh.{TriangleId, TriangleMesh3D}

import scala.collection.immutable.Seq

/**
  * Created by andreas on 8/11/18.
  */
object TFMeshOperations {

  def triangleNormals(vtx: Output, triangles: Output) = {
    val vtxsPerTriangle = tf.gather(vtx, triangles)

    val pt1 = vtxsPerTriangle(::, 0)
    val pt2 = vtxsPerTriangle(::, 1)
    val pt3 = vtxsPerTriangle(::, 2)

    val u = pt2 - pt1
    val v = pt3 - pt1

    println("u", u)

    val cross = tf.cross(u,v)

    tf.l2Normalize(cross, axes=Seq(1))
  }

  def vertexNormals(cellNormals: Output, triangleIdsForPoint: Output) = {
    println("cellNromals", cellNormals)
    println("triangleIdsForPoint", triangleIdsForPoint)
    val normalsPerVertex = tf.gather(cellNormals, triangleIdsForPoint)

    val validEntries = triangleIdsForPoint > -1f

    val sumValidEntries = tf.countNonZero(validEntries, axes=Seq(1))

    println("validEntries", validEntries)
    println("sumValidEntries", sumValidEntries)

    println("normalsPerVertex", normalsPerVertex)
    tf.l2Normalize(
      tf.sum(normalsPerVertex*tf.tile(tf.expandDims(validEntries, 2), Shape(1,1,3)), axes = Seq(1)),
      axes= Seq(1)
    )
  }


  def vertexNormals(vtx: Output, triangles: Output,  triangleIdsForPoint: Output): Output = {
    val cellNormals = triangleNormals(vtx, triangles)
    vertexNormals(cellNormals, triangleIdsForPoint)
  }

  def trianglesForPoint(data: IndexedSeq[(PointId, IndexedSeq[TriangleId])]) = {
    val sorted = data.toIndexedSeq.sortBy(_._1.id)
    val maxNeighbouringTriangles = 8
    val listOfTensors = sorted.map { case (ptid, triangles) =>
      val space = Array.fill(maxNeighbouringTriangles)(-1)
      var i = 0
      for (t <- triangles) {
        space(i) = t.id
        i += 1
      }
      Tensor(space)
    }
    Tensor(listOfTensors).reshape(Shape(data.length, maxNeighbouringTriangles))
  }

  def adjacentPoints(mesh: TriangleMesh3D) = {
    val data = mesh.triangulation.pointIds.map { ptId =>
      val adj = mesh.triangulation.adjacentPointsForPoint(ptId)
      (ptId, adj)
    }
    val sorted = data.toIndexedSeq.sortBy(_._1.id)
    val maxNeighs = 8
    val listOfTensors = sorted.map { case (ptid, neighs) =>
      val space = Array.fill(maxNeighs)(-1)
      var i = 0
      for (n <- neighs) {
        space(i) = n.id
        i += 1
      }
      Tensor(space)
    }
    Tensor(listOfTensors).reshape(Shape(data.length, maxNeighs))
  }

  /** adjacentPoints: #points X maximum possible adjacent points
    * point data:     #points X data dim*/
  def vertexToNeighbourDistance(adjacentPoints: Output, pointData: Output): Output = {
    println("adjacentPoints", adjacentPoints)
    println("pointData", pointData)

    val neighValuesPerVertex = tf.gather(pointData, adjacentPoints)
    println("neighValuesPerVertex", neighValuesPerVertex)
    val vertexTiled = tf.tile(tf.expandDims(pointData, 1), Seq(1, adjacentPoints.shape(1), 1))
    println("vertexTiled", vertexTiled)
    val neighsToVertex = tf.subtract(vertexTiled, neighValuesPerVertex)
    println("neighsToVertex", neighsToVertex)

    val validEntries = adjacentPoints > -1f
    println("validEntries", validEntries)
    //val sumValidEntries = tf.countNonZero(validEntries, axes=Seq(1))
    val validEntriesTiled = tf.tile(tf.expandDims(validEntries, 2), Seq(1, 1, pointData.shape(1)))
    println("validEntriesTiled", validEntriesTiled)

    val validDifferences = neighsToVertex * validEntriesTiled
    println("validDifferences", validDifferences)
    val res = tf.sum(tf.abs(validDifferences))
    println("res", res)
    res
  }

  def main(args: Array[String]): Unit = {
    val mesh = MeshIO.read(new File("/home/andreas/export/mean2012_l7_bfm_pascaltex.msh.gz")).get.colorNormalMesh3D.get

    val triForPoint = mesh.shape.pointSet.pointIds.toIndexedSeq.map { id =>
      (id, mesh.shape.triangulation.adjacentTrianglesForPoint(id))
    }

    val trianglesForPointData = trianglesForPoint(triForPoint)

    println("trianglesForPointData", trianglesForPointData)

    val triangles = TFMesh.triangulationAsTensor(mesh.shape.triangulation)
    val pts = TFConversions.pointsToTensor(mesh.shape.position.pointData)

    val cellNormals = triangleNormals(pts.transpose(), triangles)

    println("cellNormals", cellNormals.shape)

    val vtxNormals = vertexNormals(cellNormals, trianglesForPointData)

    println("vtxNormals", vtxNormals)

    val session = Session()
    val res = session.run(fetches = Seq(vtxNormals))

    val tfVtx = res(0).toTensor
    println(tfVtx(100,::).summarize())

    println(mesh.shape.vertexNormals.pointData(100))

  }
}
