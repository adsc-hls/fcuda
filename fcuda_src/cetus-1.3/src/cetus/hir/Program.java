package cetus.hir;

import cetus.analysis.DDGraph;
import cetus.exec.Driver;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
* Represents the entire program.
*/
public final class Program implements Traversable {

    private static Method class_print_method;

    private Method object_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Program.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
    * Every child is a TranslationUnit. An array list is used here in case
    * we want to do parallel parsing and need the indices.
    */
    private List<Traversable> children;

    /* Data Dependence Graph */
    private DDGraph ddgraph;

    /**
    * Make an empty program.
    */
    public Program() {
        object_print_method = class_print_method;
        children = new ArrayList<Traversable>();
    }

    /**
    * Make a program from a group of source files.
    *
    * @param files A list of strings that are source file names.
    */
    public Program(List<String> files) {
        object_print_method = class_print_method;
        children = new ArrayList<Traversable>();
        for (String f : files) {
            addTranslationUnit(new TranslationUnit(f));
        }
    }

    /**
    * Make a program from a group of source files.
    *
    * @param files An array of strings that are source file names.
    */
    public Program(String[] files) {
        object_print_method = class_print_method;
        children = new ArrayList<Traversable>();
        for (String f : files) {
            addTranslationUnit(new TranslationUnit(f));
        }
    }

    /**
    * Adds a translation unit to the program.
    *
    * @param tunit The translation unit to add.  Its parent
    * will be set to this program.
    */
    public void addTranslationUnit(TranslationUnit tunit) {
        if (tunit.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(tunit);
        tunit.setParent(this);
    }

    /**
    * Prints the entire program to a stream.
    *
    * @param p The program to print.
    * @param o The writer for the prorgram to be printed on.
    */
    public static void defaultPrint(Program p, PrintWriter o) {
        PrintTools.printlnList(p.children, o);
    }

    @Override
    public String toString() {
        StringWriter sw = new StringWriter(80000);
        print(new PrintWriter(sw));
        return sw.toString();
    }

    public List<Traversable> getChildren() {
        return children;
    }

    public Traversable getParent() {
        // a program has no parent
        return null;
    }

    /**
    * Write all translation units to their respective files.
    *
    * @throws FileNotFoundException if a file could not be opened. 
    */
    public void print() throws IOException {
        String outdir = Driver.getOptionValue("outdir");
        // make sure the output directory exists
        File dir = null;
        try {
            dir = new File(outdir);
            if (!dir.exists()) {
                if (!dir.mkdir()) {
                    throw new IOException("mkdir failed");
                }
            }
        } catch(IOException e) {
            System.err.println("cetus: could not create output directory, "+e);
            Tools.exit(1);
        } catch(SecurityException e) {
            System.err.println("cetus: could not create output directory, "+e);
            Tools.exit(1);
        }
        for (Traversable t : children) {
            ((TranslationUnit)t).print(dir);
        }
    }

    public void print(PrintWriter o) {
        if (object_print_method == null)
            return;
        try {
            object_print_method.invoke(null, new Object[] {this, o});
        } catch(IllegalAccessException e) {
            throw new InternalError();
        } catch(InvocationTargetException e) {
            throw new InternalError();
        }
    }

    public void removeChild(Traversable child) {
        int index = Tools.identityIndexOf(children, child);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        children.remove(index);
        child.setParent(null);
    }

    public void setChild(int index, Traversable t) {
        if (t.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (t instanceof TranslationUnit) {
            children.set(index, t);
            t.setParent(this);
        } else {
            throw new IllegalArgumentException();
        }
    }

    /**
    * Overrides the class print method, so that all subsequently
    * created objects will use the supplied method.
    *
    * @param m The new print method.
    */
    static public void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    /**
    * Unsupported - the parent of a program is null and may not be changed.
    */
    public void setParent(Traversable t) {
        throw new UnsupportedOperationException();
    }

    /**
    * Overrides the print method for this object only.
    *
    * @param m The new print method.
    */
    public void setPrintMethod(Method m) {
        object_print_method = m;
    }

    /**
    * Adds a Data Dependence Graph Object to this program, created by DDTDriver
    */
    public void createNewDDGraph() {
        ddgraph = new DDGraph();
    }

    /**
    * Return program data dependence graph
    */
    public DDGraph getDDGraph() {
        return ddgraph;
    }

}
