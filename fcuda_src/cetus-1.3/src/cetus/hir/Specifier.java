package cetus.hir;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;

/**
* Represents type specifiers and modifiers.
*/
public class Specifier implements Printable {

    private static HashMap<String, Specifier> spec_map =
            new HashMap<String, Specifier>(64);

  // John A. Stratton: February 2008
  // Added CUDA specifiers to the list
  // global, local, shared, constant, device, host
    private static String[] names = {
            "char", "wchar_t", "bool", "short", "int", "long", "signed",
            "unsigned", "float", "double", "void", "const", "volatile", "auto",
            "register", "static", "extern", "mutable", "inline", "virtual",
            "explicit", "&", "friend", "typedef", "private", "protected",
            "public", "__restrict", "transient", "final", "abstract", "native",
            "threadsafe", "synchronized", "strictfp", "boolean", "byte",
            "_Bool", "_Complex", "_Imaginary",
	    "__global__", "__local__", "__shared__", "__constant__", "__device__",
	    "__host__", "__attribute((aligned))"};

    public static final Specifier CHAR = new Specifier(0);
    public static final Specifier WCHAR_T = new Specifier(1);
    public static final Specifier BOOL = new Specifier(2);
    public static final Specifier SHORT = new Specifier(3);
    public static final Specifier INT = new Specifier(4);
    public static final Specifier LONG = new Specifier(5);
    public static final Specifier SIGNED = new Specifier(6);
    public static final Specifier UNSIGNED = new Specifier(7);
    public static final Specifier FLOAT = new Specifier(8);
    public static final Specifier DOUBLE = new Specifier(9);
    public static final Specifier VOID = new Specifier(10);

    public static final Specifier CONST = new Specifier(11);
    public static final Specifier VOLATILE = new Specifier(12);

    public static final Specifier AUTO = new Specifier(13);
    public static final Specifier REGISTER = new Specifier(14);
    public static final Specifier STATIC = new Specifier(15);
    public static final Specifier EXTERN = new Specifier(16);
    public static final Specifier MUTABLE = new Specifier(17);

    public static final Specifier INLINE = new Specifier(18);
    public static final Specifier VIRTUAL = new Specifier(19);
    public static final Specifier EXPLICIT = new Specifier(20);

    public static final Specifier REFERENCE = new Specifier(21);

    public static final Specifier FRIEND = new Specifier(22);
    public static final Specifier TYPEDEF = new Specifier(23);

    public static final Specifier PRIVATE = new Specifier(24);
    public static final Specifier PROTECTED = new Specifier(25);
    public static final Specifier PUBLIC = new Specifier(26);

    public static final Specifier RESTRICT = new Specifier(27);

    /* The following are Java-specific */

    public static final Specifier TRANSIENT = new Specifier(28);
    public static final Specifier FINAL = new Specifier(29);
    public static final Specifier ABSTRACT = new Specifier(30);

    public static final Specifier NATIVE = new Specifier(31);
    public static final Specifier THREADSAFE = new Specifier(32);
    public static final Specifier SYNCHRONIZED = new Specifier(33);

    public static final Specifier STRICTFP = new Specifier(34);

    public static final Specifier BOOLEAN = new Specifier(35);
    public static final Specifier BYTE = new Specifier(36);

    /* The following type-specifiers are added to support C99 keywords */

    public static final Specifier CBOOL = new Specifier(37);
    public static final Specifier CCOMPLEX = new Specifier(38);
    public static final Specifier CIMAGINARY = new Specifier(39);


  /* John A. Stratton: February 2008
   * The following are CUDA-specific keywords */

  public static final Specifier GLOBAL 		= new Specifier(40);
  public static final Specifier LOCAL 		= new Specifier(41);
  public static final Specifier SHARED 		= new Specifier(42);
  public static final Specifier CONSTANT	= new Specifier(43);
  public static final Specifier DEVICE 		= new Specifier(44);
  public static final Specifier HOST 		= new Specifier(45);

  /* John A. Stratton: May 2011
   * added support for gcc alignment */
  public static final Specifier ALIGNED		= new Specifier(46);
 

    /** Predefined integer value of each specifiers. */
    protected int value;

    /** Base constructor */
    protected Specifier() {
        value = -1;
    }

    private Specifier(int value) {
        this.value = value;
        spec_map.put(names[value], this);
    }

    /** Creates a specifier from the specified string. */
    public static Specifier fromString(String s) {
        return spec_map.get(s);
    }

    /** Prints the specifier to the print writer. */
    public void print(PrintWriter o) {
        if (value >= 0) {
            o.print(names[value]);
        }
    }

    /** Returns a string representation of the specifier. */
    @Override
    public String toString() {
        StringWriter sw = new StringWriter(16);
        print(new PrintWriter(sw));
        return sw.toString();
    }

    /** Checks if the specifier is used as a type specifier. */
    public boolean isCType() {
        return (value <= 10 && value >= 0) ? true : false;
    }

}
