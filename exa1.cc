template <int dim>
struct PointHistory
{
//	J2Material material_point = J2Material(/*H_=*/2/3 * 4, /*beta_=*/1.0, /*sy=*/1.2, /*E_=*/, /*nu_=*/0.3);
	double EEE = 50.0 / 3.0 / (1 - 2 * 0.3);
	DruckerPrager material_point = DruckerPrager(0.3, EEE, 30, 20000000000000, 10, 20);
//      Point<dim> point;
	double strain_xx;
		double strain_yy;