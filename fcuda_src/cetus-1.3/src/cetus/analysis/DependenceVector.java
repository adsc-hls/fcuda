package cetus.analysis;

import cetus.hir.Loop;

import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Set;

/**
 * Stores and manipulates direction vectors for loop-based dependences
 */
public class DependenceVector {
    static String[] depstr = {"*", "<", "=", ">"};
    static final int nil = -1;
    static final int any = 0;
    static final int less = 1;
    static final int equal = 2;
    static final int greater = 3;
    int cartesian_prod[][] = {
            {any, less, equal, greater},
            {less, less, nil, nil},
            {equal, nil, equal, nil},
            {greater, nil, nil, greater}};
    /* LinkedHashMap maintains ordering of loops within the vector map */
    LinkedHashMap<Loop, Integer> directionVector;
    private boolean valid = true;

    public DependenceVector(LinkedList <Loop> nest) {
        this.valid = true;
        directionVector = new LinkedHashMap<Loop,Integer>();
        for (Loop loop: nest) {
            directionVector.put(loop, 0); //any value
        }
    }
    
    public DependenceVector(DependenceVector dv) {
        this.copyVector (dv);
    }

    public boolean equals(Object o) {
        return o != null &&
                o instanceof DependenceVector &&
                directionVector.equals(((DependenceVector)o).directionVector);
    }

    public LinkedHashMap<Loop,Integer> getDirectionVector() {
        return directionVector;
    }
    
    public int getDirection (Loop loop) {
        return directionVector.get(loop);
    }
    
    public Set<Loop> getLoops() {
        return directionVector.keySet();
    }
    
    public void setDirection(Loop loop, int direction) {
        directionVector.put(loop,direction);
    }
    
    public boolean isValid() {
        return this.valid;
    }

    public void setValid(boolean value) {
        this.valid = value;
    }

    public void copyVector(DependenceVector dv) {
        this.valid = dv.valid;
        directionVector = new LinkedHashMap<Loop,Integer>();
        for (Loop loop: dv.getLoops()) {
            directionVector.put(loop, dv.getDirection(loop));
        }
    }

    public void mergeWith(DependenceVector other_vector) {
        int new_dir;
        for (Loop l : other_vector.getLoops()) {
            if (directionVector.containsKey(l)) {
                int this_dir = this.getDirection(l);
                int that_dir = other_vector.getDirection(l);
                if (this_dir != DependenceVector.nil) {
                    new_dir = cartesian_prod[this_dir][that_dir];
                } else {
                    new_dir = DependenceVector.nil;
                }
                if (new_dir == DependenceVector.nil) valid = false;
                this.directionVector.put(l, new_dir);
            } else {
                this.directionVector.put(l, other_vector.getDirection(l));
            }
        }
    }
    
    public boolean plausibleVector() {
        boolean vectorValid = true;
        Set<Loop> loopNest = this.directionVector.keySet();
        for (Loop loop : loopNest) {
            //
            // Following invalid possibilities:
            // (>,...) , (=,=,>,...) , (*,>,...)
            //
            if (this.directionVector.get(loop) == DependenceVector.greater) {
                vectorValid = false;
                break;
            }
            //
            // Else if following valid possibilities:
            // (<,...) , (=,<,...) , (*,<,...)
            //
            else if (this.directionVector.get(loop) == DependenceVector.less) {
                vectorValid = true;
                break;
            }
            //
            // Else we need to further traverse the directionVector
            // (=,...) , (*,...)
            //
            else {
            }
        }
        return vectorValid;
    }
    
    public DependenceVector reverseVector() {
        DependenceVector newDV = new DependenceVector(this);
        Set<Loop> loopKey = (this.directionVector).keySet();
        for(Loop loop : loopKey) {
            switch(this.directionVector.get(loop)) {
            case DependenceVector.any:
            case DependenceVector.equal:
            case DependenceVector.nil:
                break;
            case DependenceVector.less:
                newDV.setDirection(loop, DependenceVector.greater);
                break;
            case DependenceVector.greater:
                newDV.setDirection(loop, DependenceVector.less);
                break;
            }
        }
        return newDV;
    }
    
    public String VectorToString() {
        if (this.valid) {
            String dirvecstr = new String();
            Set<Loop> nest = this.directionVector.keySet();
            for (Loop loop: nest) {
                if(directionVector.get(loop) >= 0) {
                    dirvecstr += this.depstr[directionVector.get(loop)];
                }
            }
            return dirvecstr;
        } else {
            return ".";
        }
    }

    public String toString() {
        return VectorToString();
    }
}
