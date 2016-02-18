package cetus.analysis;

import cetus.hir.InlineAnnotation;

/**
 * 
 * Parses the inline annotations. Examples of inline annotation are:
 * #pragma inlinein
 * #pragma inline
 * #pragma noinlinein
 * #pragma noinline
 *
 */
public class InlineAnnotationParser {
	
		public static boolean parse_pragma(String [] str_array)
	{
		if(str_array.length >= 3) {
			String option = str_array[2];
			if(option.equals(InlineAnnotation.INLINE_IN_TEXT) || option.equals(InlineAnnotation.NO_INLINE_IN_TEXT) 
					|| option.equals(InlineAnnotation.INLINE_TEXT) || option.equals(InlineAnnotation.NO_INLINE_TEXT)) {
				return true;
			}	
		}
		return false;
	}	
}
