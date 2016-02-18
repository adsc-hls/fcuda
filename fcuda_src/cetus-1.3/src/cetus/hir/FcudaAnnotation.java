package cetus.hir;

import java.util.*;

/**
 * FcudaAnnotation is used for user annotations inserted by the FCUDA programmer. 
 */
public class FcudaAnnotation extends PragmaAnnotation
{
        private static final long serialVersionUID = 1; /* avoids gcc 4.3 warning */	
	/**
	 * Constructs an empty cetus annotation.
	 */
	public FcudaAnnotation()
	{
		super();
	}

	/**
	 * Constructs a cetus annotation with the given key-value pair.
	 */
	public FcudaAnnotation(String key, Object value)
	{
		super();
		put(key, value);
	}

	/**
	 * Returns a string representation of this Fcuda annotation.
	 * @return a string representation.
	 */
	public String toString()
	{
		if ( skip_print )
			return "";

		StringBuilder str = new StringBuilder(80);
		str.append(super.toString()+"fcuda "+get("fcuda")+" ");

		for ( String key : keySet() )
		{
			if (!key.equals("fcuda")) {
				if(key.equals("comp") || key.equals("trans")) {
					str.append(key + "=(");
					str.append(Tools.collectionToString((Collection<?>)this.get(key), ","));
					str.append(") ");
				}
				else			
					str.append(key+"="+get(key)+ " ");
			}
		}
			
		return str.toString();
	}

}
