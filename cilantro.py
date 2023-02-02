from pydoc import resolve
import shlex
from os import mkdir
from subprocess import  PIPE, Popen
from trace import Trace
from turtle import pu
import trimesh
import os
from copy import deepcopy
import re
from payafusion.unzip import random_filename
from trimesh import creation, transformations
import math
import open3d as o3d
import numpy as np
import json
import time
from time import process_time
import os
from os import mkdir
import sys
from turtle import width
import collada
from sklearn.preprocessing import scale
import trimesh
import numpy as np
from scipy.interpolate import splprep, splev
import math
from trimesh import creation, transformations
from payafusion.align_utils import *
import pandas as pd


DEBAG = True


 
def cilantro (dataDir,userid, userSex,dirname):
  print("______________________________ Start of Cilantro ______________________________")
  t1 = time.time()
  meshFilename = dirname + '/mesh/generated.ply'
  movingObjFilename = f'{dataDir}/model.obj'
 

  
  if dirname != '.':
    try:            
         mkdir(dirname + '/cilantro/')
    except:
            pass 

  outputDir = os.path.dirname(os.path.realpath(dirname)+ '/cilantro/')

  
  res = trimesh.visual.resolvers.FilePathResolver(movingObjFilename)
  movingMesh_model = trimesh.load(movingObjFilename, process=False, resolver=res)

   
  fixedMesh = trimesh.load_mesh(meshFilename, process=False)
  
  i = 0
  movingMesh = 0
  values = []
  keys = []
  lengths = []


  for key, value in movingMesh_model.geometry.items():
        values.append(value)
        value._visual.material.name = key
        lengths.append(len(value.vertices))
        movingMesh += values[i]
        i += 1

  # prepare cilantro non_rigid_icp fixed and moving(template) file
  translation = fixedMesh.bounds[0][1] - movingMesh.bounds[0][1] 
  fixedMeshHeight = fixedMesh.bounds[1][1] - fixedMesh.bounds[0][1]
  movingMeshHeight = movingMesh.bounds[1][1] - movingMesh.bounds[0][1]
  scale =  fixedMeshHeight / movingMeshHeight

  rot_matrix = transformations.rotation_matrix(math.pi , [0,1,0], [0, 0, 0])
  movingMesh.apply_transform(rot_matrix)
  movingMesh.apply_translation([0,translation,0])
#   movingMesh.apply_scale([scale,scale,scale])

  align_transform,cost =trimesh.registration.mesh_other(movingMesh, fixedMesh, samples=500, scale=False, icp_first=10, icp_final=50)
  movingMesh.apply_transform(align_transform)  

  finalObjectFile = outputDir + '/processed_model.ply'
  tri_mesh = trimesh.Trimesh(vertices=movingMesh.vertices,faces=movingMesh.faces,process= False, vertex_normals= movingMesh.vertex_normals)
  tri_mesh.export(finalObjectFile)
  A = os.path.dirname(os.path.realpath(dirname)+ '/mesh/') + '/body.ply '
  B = os.path.dirname(os.path.realpath(dirname)+ '/cilantro/') + '/processed_model.ply'

  
  commandlineArguments =  A +  B + ' 20' + ' 1.5e-3'
  commandlineArguments = 'non_rigid_icp ' + commandlineArguments    
  commandlineArguments = shlex.split(commandlineArguments)
  bcpdExeFilename = os.path.dirname(os.path.realpath(__file__)) + '/non_rigid_icp'
  
 
  #runing non_rigid_cilantro

  with Popen(commandlineArguments, executable=bcpdExeFilename, stdout=PIPE, bufsize=1, universal_newlines=True,cwd=outputDir) as p:
    for line in p.stdout:
        print(line, end='')
  
  
  cilantro_mesh = trimesh.load_mesh(dirname + '/cilantro/res.ply', process=False)

  
  n = cilantro_mesh.vertices.view(np.ndarray)

  body_component = []
  
  l = len(lengths)
  c = 0
  for j in range(l):
        body_component.append(n[c : c + lengths[j]])
        c = c + lengths[j] 
  
  
  count = 0
  for key, value in movingMesh_model.geometry.items():
        value.vertices = body_component[count]
        count += 1

  rot_matrix = transformations.rotation_matrix(math.pi , [0,1,0], [0, 0, 0])
#   movingMesh_model.apply_transform(rot_matrix)
  finalObjectFile = f'{dataDir}/{userid}.obj'
  finalObjectFile1 = dirname + '/cilantro/new_res.ply'
  resultScene_curves = cut_curves(movingMesh_model,userid,userSex,dataDir,dirname)

  trimesh.exchange.export.export_scene(movingMesh_model, finalObjectFile, resolver=res)
  trimesh.exchange.export.export_scene(movingMesh_model, finalObjectFile1, resolver=res)
  
  # print("_____________________ cilantro finished _____________________")
  
#  ###################### add curves #################################
 
def cut_curves(movingMesh_model,userid,userSex,dataDir,dirname):
    templateFit = trimesh.load_mesh(dirname + '/fit/out.obj')
    user = movingMesh_model.geometry['MBLab_skin2']
    user2 = movingMesh_model.geometry['MBlab_generic'] + movingMesh_model.geometry['MBLab_skin2']
    
    resultScene = trimesh.Scene()
    resultScene.add_geometry(movingMesh_model , geom_name= "user")
   
    if __name__ == "__main__":
        moduleAssetsDir = os.path.realpath('./payafusion/')
    else:
        moduleAssetsDir = os.path.dirname(__file__) 

    def planeFromPoints(points):
      AB = points[0] - points[1]
      AC = points[0] - points[2]	
    
      normal  = np.cross(AB,AC)
      mean = ( points[0] + points[1] + points[2] + points[3] ) / 4.0
      origin = mean
      return origin,normal
    def loadTemplate(filename):
      quads = {}
      template=None

      col = collada.Collada(filename, ignore=[collada.DaeUnsupportedError,
                                            collada.DaeBrokenRefError])
      if col.scene is not None:
          for geom in col.scene.objects('geometry'):
              name = geom.original.id.split('-')[0]
                
              for prim in geom.primitives():            
                
                  indices = np.array(prim.vertex_index)
                  indicesShape = indices.shape
                # faces = indices.reshape(3,indicesShape[0]*indicesShape[1])
                
                  faces = indices
                  tri_mesh = trimesh.Trimesh(vertices=prim.vertex,faces=faces)

                  if name.lower() == 'template':
                      template = tri_mesh
                  else:
                      quads[name.lower()] = tri_mesh


      return template,quads
  
    template,quads = loadTemplate( moduleAssetsDir + '/template/template_'+userSex+'.dae')
    templateCuts = {}
    templateCutPlanes = {}

    def findNearestPath(cutPath,origin):
      nearestPath = None
      minDistance = sys.float_info.max     
      for part in cutPath.discrete:            
          partPath = trimesh.load_path(part)

          dist = np.linalg.norm(partPath.centroid - origin)
          if dist<minDistance:
              minDistance  = dist
              nearestPath = partPath
      return nearestPath

    for quadName,quadMesh in quads.items():
        origin,normal = planeFromPoints(quadMesh.vertices)
        templateCutPlanes[quadName] = (origin,normal)

    chestCutPlane = templateCutPlanes['chest']
    waistCutPlane = templateCutPlanes['waist']
    hipCutPlane = templateCutPlanes['hip']
    abdomenCutPlane = templateCutPlanes['stomach']

    CX_planes = dividePlane(chestCutPlane,waistCutPlane,10)
    WX_planes = dividePlane(waistCutPlane,abdomenCutPlane,4)
    HX_planes = dividePlane(abdomenCutPlane,hipCutPlane,9)

    #TODO: add new generated planes to templateCutPlanes list
    for i in range(len(CX_planes)):
        templateCutPlanes[ "C"+ str(i)]  = CX_planes[i]

    for i in range(len(WX_planes)):
        templateCutPlanes[ "W"+ str(i)]  = WX_planes[i]

    for i in range(len(HX_planes)):
        templateCutPlanes[ "H"+ str(i)]  = HX_planes[i]


    def findBarycentricCoords(template,points):
      closest,distance,triangleIds = trimesh.proximity.closest_point(template,points)

      triangles = template.triangles[triangleIds]
      barycoords = trimesh.triangles.points_to_barycentric(triangles, closest, method='cramer')
      return triangleIds,barycoords
    for planeName,plane in templateCutPlanes.items():
      origin = plane[0]
      normal = plane[1]        
      
      lines = trimesh.intersections.mesh_plane(template,normal,origin)
      cutPath = trimesh.load_path(lines)
      
      if planeName != 'thigh' and planeName != 'calf':        
          cutPath = findNearestPath(cutPath,origin)

      trianglesIds, barycoords = findBarycentricCoords(template,cutPath.discrete[0])
      templateCuts[planeName] = (trianglesIds,barycoords)

    def getCutPaths(cutCoords):
      templateTrianglesIdxs = cutCoords[0]
      templateTrianglesBaryCoords = cutCoords[1]

      corrTriangles = templateFit.triangles[templateTrianglesIdxs]
      corrPoints = trimesh.triangles.barycentric_to_points(corrTriangles,templateTrianglesBaryCoords)

      corrCutPlane = trimesh.points.plane_fit(corrPoints)
      origin = corrCutPlane[0]
      normal = corrCutPlane[1]        

      lines = trimesh.intersections.mesh_plane(user,normal,origin)
      cutPath = trimesh.load_path(lines)

      return cutPath,origin,normal

    def getCutPaths_hip(cutCoords):
      templateTrianglesIdxs = cutCoords[0]
      templateTrianglesBaryCoords = cutCoords[1]

      corrTriangles = templateFit.triangles[templateTrianglesIdxs]
      corrPoints = trimesh.triangles.barycentric_to_points(corrTriangles,templateTrianglesBaryCoords)

      corrCutPlane = trimesh.points.plane_fit(corrPoints)
      origin = corrCutPlane[0]
      normal = corrCutPlane[1]        

      lines = trimesh.intersections.mesh_plane(user2,normal,origin)
      cutPath = trimesh.load_path(lines)

      return cutPath,origin,normal

    def estimateClosedCurve(points,debug=False):
      x = np.array([x for x,y,z in points])
      y = np.array([y for x,y,z in points])
      z = np.array([z for x,y,z in points])  

      tck, u = splprep([x,y,z], u=None, s=0.005, per=1,k=4) 
      u_new = np.linspace(u.min(), u.max(), 300)
      x_new, y_new, z_new = splev(u_new, tck, der=0)
      x_new = np.reshape(x_new,(len(x_new),1))
      y_new = np.reshape(y_new,(len(y_new),1))
      z_new = np.reshape(z_new,(len(z_new),1))
    
      return np.hstack([x_new,y_new,z_new])

    
    def findPlane(templateFit,cutCoords):
      templateTrianglesIdxs = cutCoords[0]
      templateTrianglesBaryCoords = cutCoords[1]

      corrTriangles = templateFit.triangles[templateTrianglesIdxs]
      corrPoints = trimesh.triangles.barycentric_to_points(corrTriangles,templateTrianglesBaryCoords)     

      corrCutPlane = trimesh.points.plane_fit(corrPoints)
      origin = corrCutPlane[0]
      normal = corrCutPlane[1]

      return (origin,normal)

    def reducePlanesFromCurves2(cutPath,rightPlane,leftPlane):
      rightOrigin,rightNormal = rightPlane
      leftOrigin,leftNormal = leftPlane

      start =0
      end = 0

      if rightOrigin[0] < leftOrigin[0]:
          start = rightOrigin[0]
          end = leftOrigin[0]
      else:
          start = leftOrigin[0]
          end = rightOrigin[0]

      def pointInside(points,start,end):
          result = []
          for p in points:
              if p[0]>start and p[0]<end:
                  result.append(1)
              else:
                  result.append(0)
          result = np.array(result)
          return result
        
      pathPoints = cutPath.discrete[0]
      r = pointInside(cutPath.discrete[0],start,end)    
      pathPoints = pathPoints[r==1 ]
    

      if pathPoints.size == 0:
          return cutPath

      curvePoints = estimateClosedCurve(pathPoints)
    
      cutPath = trimesh.load_path(curvePoints)
      return cutPath
    
    def seperateLeftAndRightPart(cutPath):
        parts = []
        for e in cutPath.discrete:
            part = trimesh.load_path(e)
            parts.append(part)
        if parts[0].centroid[0] < parts[1].centroid[0]:
            return parts[0],parts[1]
        else:                
            return parts[1],parts[0]    


#################### calculate chest curve ########################
    
    rightHandCutPlane = findPlane(templateFit,templateCuts['handcut_r'])
    leftHandCutPlane = findPlane(templateFit,templateCuts['handcut_l'])
    del templateCuts['handcut_r']
    del templateCuts['handcut_l']
    measurements_for_vis = {}
      
# chest
    if userSex == 'F':
        chestcutCurves,origin,normal = getCutPaths_hip(templateCuts['C0'])
    else:   
        chestcutCurves,origin,normal = getCutPaths(templateCuts['C0'])
    chestCurve = findNearestPath(chestcutCurves,origin) 
    chestCurve = reducePlanesFromCurves2(chestCurve,rightHandCutPlane,leftHandCutPlane)  
    measurements_for_vis['chest'] = toCentimeter(chestCurve.length)
    resultScene.add_geometry(chestCurve, geom_name='chest')


# Wrist 

    wrist_r_cutCurves,origin,normal = getCutPaths(templateCuts['wrist_r'])
    wrist_r_Curve = findNearestPath(wrist_r_cutCurves,origin)
  
    rightWristLength =toCentimeter(wrist_r_Curve.length)
    measurements_for_vis['wrist_r'] = toCentimeter(wrist_r_Curve.length)
    
    wrist_l_cutCurves,origin,normal = getCutPaths(templateCuts['wrist_l'])
    wrist_l_Curve = findNearestPath(wrist_l_cutCurves,origin)
    leftWristLength = toCentimeter(wrist_l_Curve.length)
    measurements_for_vis['wrist_l'] = toCentimeter(wrist_l_Curve.length)

    if (rightWristLength < 25) & (leftWristLength < 25):
        resultScene.add_geometry(wrist_r_Curve, geom_name='wrist_r')
        resultScene.add_geometry(wrist_l_Curve, geom_name='wrist_l')


# neck

    neckcutCurves,origin,normal = getCutPaths(templateCuts['neck'])
    neckCurve = findNearestPath(neckcutCurves,origin)
    resultScene.add_geometry(neckCurve, geom_name='neck')
    measurements_for_vis['neck'] = toCentimeter(neckCurve.length)


# waist 

    waistCurve,origin,_ = getCutPaths(templateCuts['W0'])
    waistCurve = findNearestPath(waistCurve,origin) 
    # waistCurve.colors = [(255,0,0,255)] 
    resultScene.add_geometry(waistCurve, geom_name='waist')
    measurements_for_vis['waist'] = toCentimeter(waistCurve.length)
# hip 

    hipCurve,origin,_ = getCutPaths_hip(templateCuts['H8'])
    hipCurve = findNearestPath(hipCurve,origin)    
    # hipCurve.colors = [(255,0,0,255)]
    resultScene.add_geometry(hipCurve, geom_name='hip')
    measurements_for_vis['hip'] = toCentimeter(hipCurve.length)


# stomach 
    
    if userSex == 'F':
        cutCurves,origin,normal = getCutPaths_hip(templateCuts['stomach'])
    else:   
        cutCurves,origin,normal = getCutPaths(templateCuts['stomach'])

    # cutCurves,origin,normal = getCutPaths(templateCuts['stomach'])
    abdomencurve = findNearestPath(cutCurves,origin)   
    # abdomencurve.colors = [(255,0,0,255)]
    resultScene.add_geometry(abdomencurve, geom_name='stomach')
    measurements_for_vis['stomach'] = toCentimeter(abdomencurve.length)

# upperarm 

    cutCurves,origin,normal = getCutPaths(templateCuts['upperarm_l'])
    upperarm_l_curve = findNearestPath(cutCurves,origin)
    resultScene.add_geometry(upperarm_l_curve, geom_name='upperarm_l')
    measurements_for_vis['upperarm_l'] = toCentimeter(upperarm_l_curve.length)


    cutCurves,origin,normal = getCutPaths(templateCuts['upperarm_r'])
    upperarm_r_curve = findNearestPath(cutCurves,origin)
    resultScene.add_geometry(upperarm_r_curve, geom_name='upperarm_r')
    measurements_for_vis['upperarm_r'] = toCentimeter(upperarm_r_curve.length)


#################### forearm ####################################    

    cutCurves,origin,normal = getCutPaths(templateCuts['forearm_l'])
    forearm_l_curve = findNearestPath(cutCurves,origin)
    resultScene.add_geometry(forearm_l_curve, geom_name='forearm_l')
    measurements_for_vis['forearm_l'] = toCentimeter(forearm_l_curve.length)


    cutCurves,origin,normal = getCutPaths(templateCuts['forearm_r'])
    forearm_r_curve = findNearestPath(cutCurves,origin)
    resultScene.add_geometry(forearm_r_curve, geom_name='forearm_r')
    measurements_for_vis['forearm_r'] = toCentimeter(forearm_r_curve.length)

#################### split Into Left And Right ###################   

    def splitIntoLeftAndRight(curve):
      points = np.array(cutCurves.discrete[0],dtype=np.float32 )
      max = np.max(points,axis=0)
      min = np.max(points,axis=0)
      mean = np.mean(points,axis=0)
      leftPoints = []
      rightPoints = []
      for p in points:
          if p[0] < mean[0]:
              leftPoints.append(p)
      else:
            rightPoints.append(p)
      leftCurve = trimesh.load_path(estimateClosedCurve(leftPoints))
      rightCurve = trimesh.load_path(estimateClosedCurve(rightPoints))

      return leftCurve,rightCurve

#################### thigh #######################################

    cutCurves,origin,normal = getCutPaths(templateCuts['thigh']) 

    if len(cutCurves.discrete) == 1:        
        thigh_l_curve,thigh_r_curve = splitIntoLeftAndRight(cutCurves)
    else:
        thigh_l_curve,thigh_r_curve = seperateLeftAndRightPart(cutCurves)

    resultScene.add_geometry(thigh_r_curve, geom_name='thigh_r')
    resultScene.add_geometry(thigh_l_curve, geom_name='thigh_l')

    measurements_for_vis['thigh_l'] = toCentimeter(thigh_l_curve.length)
    measurements_for_vis['thigh_r'] = toCentimeter(thigh_r_curve.length)
#################### calf ######################################### 

    cutCurves,origin,normal = getCutPaths(templateCuts['calf'])    
    calf_l_curve,calf_r_curve = seperateLeftAndRightPart(cutCurves)

    resultScene.add_geometry(calf_r_curve, geom_name='calf_r')
    resultScene.add_geometry(calf_l_curve, geom_name='calf_l')  

    measurements_for_vis['calf_l'] = toCentimeter(calf_l_curve.length)
    measurements_for_vis['calf_r'] = toCentimeter(calf_r_curve.length)

    vis_df = pd.DataFrame(measurements_for_vis,[0])        
    vis_df.to_csv(dirname + '/cilantro/out_vis_cilantro.csv')
    vis_df.to_excel(dirname + '/cilantro/out_vis_cilantro.xlsx')
    
    if DEBAG:
        resultScene.show()  
 
    mesh_path_glb = f'{dataDir}/{userid}_curves.glb'
    resultScene.export(mesh_path_glb)

    return resultScene

# #  ###########################################################################################################################
# # animation
#   meshFilename = dirname + '/mesh/generated.ply'
#   fixedMesh = trimesh.load_mesh(meshFilename, process=False)

#   movingObjFilename = '/home/saleh/TestHoma/input/test.obj'
#   outputDir = os.path.dirname(os.path.realpath(dirname)+ '/cilantro/')
#   movingMesh = trimesh.load(movingObjFilename, process=False) 

#   translation = fixedMesh.bounds[0][1] - movingMesh.bounds[0][1] 
#   fixedMeshHeight = fixedMesh.bounds[1][1] - fixedMesh.bounds[0][1]
#   movingMeshHeight = movingMesh.bounds[1][1] - movingMesh.bounds[0][1]
#   scale =  fixedMeshHeight / movingMeshHeight

#   rot_matrix = transformations.rotation_matrix(math.pi , [0,1,0], [0, 0, 0])
#   movingMesh.apply_transform(rot_matrix)
#   movingMesh.apply_translation([0,translation,0])
#   movingMesh.apply_scale([scale,scale,scale])

#   finalObjectFile = outputDir + '/processed_model.ply'
#   tri_mesh = trimesh.PointCloud(vertices=movingMesh.vertices,process= False)
#   tri_mesh.export(finalObjectFile)

#   A = os.path.dirname(os.path.realpath(dirname)+ '/mesh/') + '/body.ply '
#   B = os.path.dirname(os.path.realpath(dirname)+ '/cilantro/') + '/processed_model.ply'

  
#   commandlineArguments =  A +  B + ' 20' + ' 1.5e-3'
#   commandlineArguments = 'non_rigid_icp ' + commandlineArguments    
#   commandlineArguments = shlex.split(commandlineArguments)
#   bcpdExeFilename = os.path.dirname(os.path.realpath(__file__)) + '/non_rigid_icp'
  
 
#   #runing non_rigid_cilantro

#   with Popen(commandlineArguments, executable=bcpdExeFilename, stdout=PIPE, bufsize=1, universal_newlines=True,cwd=outputDir) as p:
#     for line in p.stdout:
#         print(line, end='')
  
#   cilantro_mesh = trimesh.load_mesh(dirname + '/cilantro/res.ply', process=False)
#   rot_matrix = transformations.rotation_matrix(math.pi , [0,1,0], [0, 0, 0])
#   cilantro_mesh.apply_transform(rot_matrix)
#   finalObjectFile = dirname + '/cilantro/new_res.ply'
#   cilantro_mesh.export(finalObjectFile)