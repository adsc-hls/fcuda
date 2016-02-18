package cetus.analysis;

/**
 * Class containing all functionalities for a weak greatest
 * common divisor (GCD) test.
 */
public class GCD
{
	/**
	 * Computes the GCD of the two given numbers.
	 */
	public static int compute(int a, int b)
	{
		a = Math.abs(a);
		b = Math.abs(b);

		for (;;)
		{
			if (a == 0)
				return b;

			if (b == 0)
				return a;

			if (a > b)
				a %= b;
			else      
				b %= a;
		}
	}

	/**
	 * @param a    All coefficients in the first expression.
	 * @param b    All coefficients in the second expression.
	 * @param c    The constant part of the equation.
	 * @param nest The size of a and b.
	 *
	 * @return false if there appears to be a dependence or
	 *         true otherwise
	 */
	public static boolean test(int[] a, int[] b, int c, int nest)
	{
		int gcd = 0;

		/* Computes GCD of multiple numbers using Euclid's algorithm. */
		for (int i = 0; i < nest; i++)
		{
			gcd = compute(gcd, a[i]);
			gcd = compute(gcd, b[i]);
		}

		if (gcd == 0)
			return false; 

		return (c % gcd != 0);
	}

	public static long[] computeWithLinearCombination(long a, long b)
	{
		long triplet[] = new long[3];
		a = Math.abs(a);
		b = Math.abs(b);
		long quotient, temp;
		long x = 0, lastx = 1;
		long y = 1, lasty = 0;
		
		while (b != 0)
		{
			quotient = a/b;
			temp = b;
			b = a % b;
			a = temp;
			
			temp = x;
			x = lastx - (quotient)*x;
			lastx = temp;
			
			temp = y;
			y = lasty - (quotient)*y;
			lasty = temp;
		}
		
		triplet[0] = a; //gcd
		triplet[1] = lastx; //x
		triplet[2] = lasty; //y
		return triplet;		
	}	
}
