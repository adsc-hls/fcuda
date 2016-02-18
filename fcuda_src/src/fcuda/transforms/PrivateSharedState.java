package fcuda.transforms;  
import fcuda.*;

import java.util.*;

import cetus.hir.*;

public class PrivateSharedState implements BlockStateTransform 
{
  public PrivateSharedState()
  { 
  }

  public void TransformSharedVars(List<VariableDeclaration> vDecls)
  {
    return;
  }

}   
