package fcuda.transforms;
import fcuda.*;

import cetus.hir.*;
import cetus.transforms.*;

import java.util.*;

public class EditHeaders extends TransformPass
{
  static Set<String> MCUDAReplacements;

  static {
    MCUDAReplacements = new HashSet<String>();
    MCUDAReplacements.add("cutil.h");
    MCUDAReplacements.add("cuda.h");
    MCUDAReplacements.add("builtin_types.h");
    MCUDAReplacements.add("cuda_runtime_api.h");
    MCUDAReplacements.add("cuda_runtime.h");
  }

  public EditHeaders(Program program)
  {
    super(program);
  }

  public String getPassName()
  {
    return new String("[EditHeaders]");
  }

  public void start()
  {
    ReplaceWithOne(program, MCUDAReplacements, "mcuda.h");
  }

  public static void ReplaceAll(Traversable root, Map<String,String> replacementMap)
  {
    DepthFirstIterator iter = new DepthFirstIterator(root);
    iter.pruneOn(Expression.class);

    while(iter.hasNext())
    {
      AnnotationDeclaration a;
      //*cetus-1.1*   try{ a = iter.next(AnnotationDeclaration.class); } 
      try{ a = (AnnotationDeclaration)iter.next(AnnotationDeclaration.class); } 
      catch(NoSuchElementException e) { break; }

      List<PragmaAnnotation> plist = a.getAnnotations(PragmaAnnotation.class);

      for(PragmaAnnotation pa : plist) 
      {
        String pragma = pa.get("pragma");

        if ( pragma.startsWith(" startinclude") )
        {
          for(String old : replacementMap.keySet())
          {
            if(pragma.contains(old)) {
              pa.put("pragma", pragma.replace(old,replacementMap.get(old)));
              break;
            }
          }
        }
      }
    }
  }

  public static void ReplaceWithOne(Traversable root, Set<String> old, String replace)
  {
    DepthFirstIterator iter = new DepthFirstIterator(root);
    iter.pruneOn(Expression.class);

    while(iter.hasNext())
    {
      AnnotationDeclaration a;
      //*cetus-1.1*   try{ a = iter.next(AnnotationDeclaration.class); } 
      try{ a = (AnnotationDeclaration)iter.next(AnnotationDeclaration.class); } 
      catch(NoSuchElementException e) { break; }

      List<PragmaAnnotation> plist = a.getAnnotations(PragmaAnnotation.class);

      for(PragmaAnnotation pa : plist) 
      {
        String pragma = pa.get("pragma");

        if ( pragma.startsWith(" startinclude") )
        {
          for(String oldheader : old)
          {
            if(pragma.contains("\"" + oldheader) || 
                pragma.contains("<" + oldheader)) {
              if(replace != null) {
                pa.put("pragma", pragma.replace(oldheader,replace));
                replace = null;
              }
              else {
                pa.put("pragma", " startinclude ");
              }
              break;
                }
          }
        }
      }
    }
  }

}

