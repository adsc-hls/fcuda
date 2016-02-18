package cetus.hir;

/**
 * 
 * Introduces inline annotations. Examples are:
 * #pragma inlinein
 * #pragma inline
 * #pragma noinlinein
 * #pragma noinline
 *
 */
public class InlineAnnotation extends PragmaAnnotation {

    private static final long serialVersionUID = 3478L;

	public static final String INLINE_IN_TEXT = "inlinein";
	public static final String NO_INLINE_IN_TEXT = "noinlinein";
	public static final String INLINE_TEXT = "inline";
	public static final String NO_INLINE_TEXT = "noinline";

	public static final InlineAnnotation INLINE_IN = new InlineAnnotation(INLINE_IN_TEXT);
	public static final InlineAnnotation NO_INLINE_IN = new InlineAnnotation(NO_INLINE_IN_TEXT);
	public static final InlineAnnotation INLINE = new InlineAnnotation(INLINE_TEXT);
	public static final InlineAnnotation NO_INLINE = new InlineAnnotation(NO_INLINE_TEXT);
	
	/**
	 * Constructs an pragma
	 */
	public InlineAnnotation(String inlineOption)
	{
		super(inlineOption);
	}
	
	/*
	 * (non-Javadoc)
	 * @see java.util.AbstractMap#equals(java.lang.Object)
	 */
	public boolean equals(Object o) {
		if(o instanceof InlineAnnotation && getName().equals(((InlineAnnotation)o).getName())) {
			return true;
		}
		return false;
	}
}
