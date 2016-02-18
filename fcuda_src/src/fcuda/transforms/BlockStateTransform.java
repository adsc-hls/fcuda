package fcuda.transforms;
import fcuda.*;

import java.util.*;
import cetus.hir.*;

/* Utility interface for abstracting away the transformations applied 
 * to block-private kernel state */
public interface BlockStateTransform 
{
  public abstract void TransformSharedVars(List<VariableDeclaration> vDecls);
}

