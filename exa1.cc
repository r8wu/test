#include <torch/script.h> 
#include <torch/torch.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <set>
#include <fstream>
#include <iostream>
#include <cmath>
#include <deal.II/numerics/solution_transfer.h>
#include <Eigen/Dense>


#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#define FORCE_USE_OF_TRILINOS
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>


using namespace dealii;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using namespace torch::indexing;

class J2Material {
public:
	double H;
	double beta;
	Vec6 alpha;
	double r;
	double e_bar;
	Vec6 ep;
	Vec6 sig;

	double E;
	double nu;
	double mu;
	double lam;
	double K;

	Mat6 Ce;
	Mat6 Ct;
	Mat6 Ct_ori;

	int yield_flag;
	
//	J2Material() : H(0), beta(0), E(0), nu(0) {}
//	

	J2Material(double H_, double beta_, double sy, double E_, double nu_) :
		H(H_), beta(beta_), E(E_), nu(nu_) {
		alpha.setZero();
		ep.setZero();
		sig.setZero();
		e_bar = 0.0;
		yield_flag = 0;

		r = std::sqrt(2.0 / 3.0) * sy;

		mu = E / (1 + nu) / 2.0;
		lam = E * nu / ((1 + nu) * (1 - 2 * nu));
		K = E / 3.0 / (1 - 2 * nu);

		Eigen::Matrix3d c1 = Eigen::Matrix3d::Constant(lam);
		c1.diagonal().array() += 2 * mu;
		Eigen::Matrix3d c2 = Eigen::Matrix3d::Identity() * mu;

		Ce.setZero();
		Ce.block<3, 3>(0, 0) = c1;
		Ce.block<3, 3>(3, 3) = c2;
		Ct = Ce;

	}

	J2Material returnMapping(const Vec6 &DEP, const std::string &response = "plastic", double dt = 0.5, double tau = 1e9) const {
		// Copy the current state
		J2Material new_state = *this;

//		Vec6 dep = DEP - ep;
//		new_state.ep += dep;
//		Vec6 sig_trial = sig + Ce * dep;
				
			
		new_state.ep += DEP;
		Vec6 sig_trial = sig + Ce * DEP;
		
		// Calculate xi        
		double p = (sig_trial(0) + sig_trial(1) + sig_trial(2)) / 3.0;
		Vec6 s_trial = sig_trial;
		s_trial.head<3>().array() -= p;
		Vec6 xi_trial = s_trial - alpha;

		// norm of xi
		double sum_val = xi_trial.head<3>().squaredNorm() + 2.0 * xi_trial.tail<3>().squaredNorm();
		double X = std::sqrt(sum_val);
		Vec6 n = xi_trial / X;

		if (X <= r) {
			new_state.yield_flag = 0;
			new_state.sig = sig_trial;
			new_state.Ct = Ce;

		} else {
			new_state.yield_flag = 1;
			double delta_lam = (X - r) / (2 * mu + H);

			new_state.sig = sig_trial - 2 * mu * delta_lam * n;
			new_state.r = r + beta * H * delta_lam;
			new_state.alpha = alpha + (1 - beta) * H * delta_lam * n;

			// cto
			double gamma_val = 2 * mu * delta_lam / X;
			Vec6 identity2;
			identity2 << 1, 1, 1, 0, 0, 0;

			Mat6 identity4 = Mat6::Zero();
			identity4(0, 0) = identity4(1, 1) = identity4(2, 2) = 1.0;
			identity4(3, 3) = identity4(4, 4) = identity4(5, 5) = 0.5;

			Mat6 outer_id = identity2 * identity2.transpose();
			Mat6 outer_n = n * n.transpose();

			Mat6 term = identity4 - (1.0 / 3.0) * outer_id - outer_n;
			Mat6 Creduct = 2 * mu * gamma_val * term;

			Mat6 Cep = K * outer_id;
			Mat6 term2 = identity4 - (1.0 / 3.0) * outer_id - outer_n / (1 + H / (2 * mu));
			Cep += 2 * mu * term2;
			Cep -= Creduct;
			new_state.Ct = Cep;

			// Viscoplasticity 
			if (response == "viscoplastic") {
				new_state.sig = (sig_trial + (dt / tau) * new_state.sig) / (1 + dt / tau);
				new_state.r = (r + dt / tau * new_state.r) / (1 + dt / tau);
				new_state.alpha = (alpha + (dt / tau) * new_state.alpha) / (1 + dt / tau);
				new_state.Ct = (Ce + (dt / tau) * new_state.Ct) / (1 + dt / tau);
			}
		}
		return new_state;
	}
};

//
//template <int dim>
//class PrescribedDisplacement : public Function<dim>
//{
//public:
//  PrescribedDisplacement() : Function<dim>(dim) {}
//  virtual void vector_value (const Point<dim> &p,
//                             Vector<double>   &values) const override
//  {
//    values(0) = 0.01;
//    for (unsigned int i=1; i<dim; ++i)
//      values(i) = 0.0;
//  }
//};
//
//  
//template <int dim>
//class RightHandSide : public Function<dim>
//{
//public:
//   virtual void vector_value(const Point<dim> &p,
//                                Vector<double>   &values) const override
//      {
//        AssertDimension(values.size(), dim);
//        Assert(dim >= 2, ExcInternalError());
//
//        Point<dim> point_1, point_2;
//        point_1[0] = 0.5;
//        point_2[0] = -0.5;
//
//        if (((p - point_1).norm_square() < 0.2 * 0.2) ||
//            ((p - point_2).norm_square() < 0.2 * 0.2))
//          values(0) = 1;
//        else
//          values(0) = 0;
//
//        if (p.square() < 0.2 * 0.2)
//          values(1) = 1;
//        else
//          values(1) = 0;
//      }
//
//      virtual void
//      vector_value_list(const std::vector<Point<dim>> &points,
//                        std::vector<Vector<double>>   &value_list) const override
//      {
//        const unsigned int n_points = points.size();
//
//        AssertDimension(value_list.size(), n_points);
//
//        for (unsigned int p = 0; p < n_points; ++p)
//          RightHandSide<dim>::vector_value(points[p], value_list[p]);
//      }
//    };
//  
//


template <int dim>
struct PointHistory
{
//	J2Material material_point = J2Material(/*H_=*/2/3 * 4, /*beta_=*/1.0, /*sy=*/1.2, /*E_=*/50.0, /*nu_=*/0.3);
	J2Material material_point = J2Material(/*H_=*/0.0, /*beta_=*/1.0, /*sy=*/1000.0, /*E_=*/10000000.0, /*nu_=*/0.3);
//      Point<dim> point;
	double strain_xx;
		double strain_yy;
		double strain_xy;
//	torch::Tensor Z_inf = torch::zeros({1, 80}, torch::TensorOptions().requires_grad(true));
	
};



 


template<int dim>
class PlasticityProblem {
public:
	PlasticityProblem();
	void run();

private:
	void setup_system();
	void assemble_system(const double current_load_factor, std::vector<std::vector<PointHistory<dim>>> &temp_ip_data);
	void solve();
	void refine_grid();
	void output_results(const unsigned int load_step) const;
	
	void get_stress_strain_tensor(
					const Eigen::Matrix<double, 6, 1> &B,
					const Eigen::Matrix<double, 6, 6> &A,
				    SymmetricTensor<2, dim>       &stress,
				    SymmetricTensor<4, dim>       &tangent);
				  
	void get_stress_strain_tensor_model(
					const torch::Tensor &stress_tensor,
					const torch::Tensor &input_tensor,
					SymmetricTensor<2, dim>       &stress,
					SymmetricTensor<4, dim>       &tangent);

	
	MPI_Comm mpi_communicator;
	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;
	parallel::distributed::Triangulation<dim> triangulation;
	IndexSet locally_owned_dofs;
	IndexSet locally_relevant_dofs;      
	ConditionalOStream pcout;
	TimerOutput        computing_timer;
	      
	      

	FESystem<dim> fe;
	DoFHandler<dim> dof_handler;
	const QGauss<dim> quadrature_formula;

	AffineConstraints<double> zero_constraints;
	AffineConstraints<double> nonzero_constraints;

	SparsityPattern sparsity_pattern;
	LA::MPI::SparseMatrix system_matrix;

	LA::MPI::Vector current_solution;
	LA::MPI::Vector previous_solution;
	LA::MPI::Vector system_rhs;
	LA::MPI::Vector newton_update;
	LA::MPI::Vector incremental_displacement;

	std::vector<std::vector<PointHistory<dim>>> quadrature_point_data;
//	std::vector<std::vector<PointHistory<dim>>> tmp;
//	std::vector<PointHistory<dim>> quadrature_point_history;
	const double tolerance;
	const unsigned int max_newton_iterations;
	const unsigned int n_load_steps;
	
	void computeConcatenatedGradients(const SymmetricTensor<2, dim> &A_previous, 
										const SymmetricTensor<2, dim> &A, 
										torch::Tensor &previous_Z,
										torch::Tensor &output_decoder,
										torch::Tensor &CTO);

	torch::jit::Module module_param = torch::jit::load("INCDE_param.pt");
		torch::jit::Module module = torch::jit::load("INCDE_Nhat.pt");
		torch::jit::Module module_decoder = torch::jit::load("INCDE_decoder.pt");
		
	int batch_size = 1;
		int t = 0;
	
};





template <int dim>
void PlasticityProblem<dim>::computeConcatenatedGradients(const SymmetricTensor<2, dim> &A_previous, 
															const SymmetricTensor<2, dim> &A, 
															torch::Tensor &previous_Z,
															torch::Tensor &output_decoder,
															torch::Tensor &CTO) {  
// auto x = torch::tensor({{A[0][0], A[1][1], A[2][2], A[1][2], A[0][2], A[0][1]}}, torch::TensorOptions().requires_grad(true)) / 2 / module_param.attr("eps_as").toTensor().item<double>(); 1x6
	
	auto x = torch::tensor({{{A[0][0], A[1][1], A[0][0]*0.0, A[0][0]*0.0, A[0][0]*0.0, A[0][1]*2.0}}}, torch::TensorOptions().requires_grad(true)) / 2 / module_param.attr("eps_as").toTensor().item<double>();
	auto x_minus_1 = torch::tensor({{{A_previous[0][0], A_previous[1][1], A_previous[0][0]*0.0, A_previous[0][0]*0.0, A_previous[0][0]*0.0, A_previous[0][1]*2.0}}}, torch::TensorOptions().requires_grad(true)) / 2 / module_param.attr("eps_as").toTensor().item<double>();
 

//	std::vector<torch::jit::IValue> inputs;
//	 inputs.push_back(module_inputs);
// torch::Tensor output = module.forward(inputs).toTensor();
 
 ///////////////////////

// auto x = x_full.index({Slice(), Slice(1, None), Slice()}).clone();
//     auto x_minus_1 = x_full.index({Slice(), Slice(None, -1), Slice()}).clone();
 //    auto x_minus_1 = x_full.index({Slice(), Slice(0, -1), Slice()}).clone();
     
 //    auto x = x_full.slice(1, 1, x_full.size(1));
 //    auto x_minus_1 = x_full.slice(1, 0, x_full.size(1)-1);
     
     auto deps = x - x_minus_1;
//     int batch_size = x_minus_1.size(0);
     
  
//     auto h_0 = torch::zeros({batch_size, module_param.attr("hidden_size").toInt()}, x.options());
	 
     
     auto h_t_minus_1 = previous_Z.clone();
     auto h_t1 = previous_Z.clone();

     std::vector<torch::Tensor> output_list;

 //    auto k_values = torch::arange(0, 1, module.dt, x.options());
//     for (int t = 0; t < x_minus_1.size(1); t++) {
     	 
 		auto deps_slice = deps.index({Slice(), Slice(t, t+1), Slice()});
 	//    	deps.slice(1, t, t+1)
 		auto deps_trans = deps_slice.transpose(1, 2);
 		
 		for (double k = 0.0; k < 1.0; k += module_param.attr("dt").toDouble()) {
 //      for (int idx_k = 0; idx_k < k_values.size(0); idx_k++) 
 //        double k = k_values[idx_k].item<double>();

//         x_minus_1.select(1, t) + k * deps.select(1, t), deps.select(1, t)
         std::vector<torch::jit::IValue> IN;	
         IN.push_back(torch::cat({h_t_minus_1, x_minus_1.index({Slice(), t, Slice()}) + k * deps.index({Slice(), t, Slice()}), deps.index({Slice(), t, Slice()})}, 1)); 
         
         auto N = module.forward(IN).toTensor().view({batch_size, module_param.attr("hidden_size").toInt(), 6});
         auto bmm_result = torch::bmm(N, deps_trans);  
         auto N_hat = (1 - h_t_minus_1.pow(2)) * bmm_result.squeeze(2);
 //        bmm_result.squeeze(-1)
         
         auto h_half = h_t1 + (module_param.attr("dt").toDouble() / 2.0) * N_hat;
         h_t_minus_1 = h_half.clone();
         
         std::vector<torch::jit::IValue> IN2;	
         IN2.push_back(torch::cat({h_t_minus_1, x_minus_1.index({Slice(), t, Slice()}) + (k + module_param.attr("dt").toDouble() / 2.0) * deps.index({Slice(), t, Slice()}), deps.index({Slice(), t, Slice()})}, 1));
      
         auto N2 = module.forward(IN2).toTensor().view({batch_size, module_param.attr("hidden_size").toInt(), 6});
         auto bmm_result2 = torch::bmm(N2, deps_trans);
         auto N_hat2 = (1 - h_t_minus_1.pow(2)) * bmm_result2.squeeze(2);

         auto h_t = h_t1 + module_param.attr("dt").toDouble() * N_hat2;
         h_t_minus_1 = h_t.clone();
         h_t1 = h_t.clone();
       }

 	
 		output_list.push_back(h_t1);
 		previous_Z = h_t1.clone();
//     } 

     auto output_tensor = torch::stack(output_list, 1);
 //    auto output_tensor = torch::stack(output, 0).transpose(0, 1);

 
  
     std::vector<torch::jit::IValue> IN_DECODER;	
     IN_DECODER.push_back(torch::cat({output_tensor, x}, 2));
     output_decoder = module_decoder.forward(IN_DECODER).toTensor().squeeze(0);
 
     output_decoder.slice(1, 0, 3) *= (2 * module_param.attr("sig_a").toTensor().item<double>());
	 output_decoder.slice(1, 3, 6) *= (2 * module_param.attr("sig_s").toTensor().item<double>());
	
	
 //////////////////////
 std::vector<torch::Tensor> gradient_list;
 int num_elements = output_decoder.numel();   

 for (int i = 0; i < num_elements; ++i) {
	 torch::Tensor grad_output = torch::ones_like(output_decoder.index({0,i}));

	 auto gradient = torch::autograd::grad({output_decoder.index({0,i})}, {x}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true);
	 gradient_list.push_back(gradient[0].squeeze(0)); 
 }

 CTO = torch::cat(gradient_list, /*dim=*/0); 

}

template <int dim>
void PlasticityProblem<dim>::get_stress_strain_tensor_model(
	const torch::Tensor &stress_tensor,
	const torch::Tensor &input_tensor,
    SymmetricTensor<2, dim>       &stress,
    SymmetricTensor<4, dim>       &tangent)
{
		if (dim == 3) {
			// Define mapping for 3D (Voigt notation)
			auto voigt_index = [](int i, int j) -> int {
				if (i > j)
					std::swap(i, j);
				if (i == 0 && j == 0)
					return 0;
				else if (i == 1 && j == 1)
					return 1;
				else if (i == 2 && j == 2)
					return 2;
				else if (i == 1 && j == 2)
					return 3;
				else if (i == 0 && j == 2)
					return 4;
				else if (i == 0 && j == 1)
					return 5;
				throw std::logic_error("Invalid indices for Voigt mapping");
			};

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					
					int I = voigt_index(i, j);
					stress[i][j] = stress_tensor.index( {0, I}).item<double>();
					
					for (int k = 0; k < 3; ++k) {
						for (int l = 0; l < 3; ++l)  {
							int J = voigt_index(k, l);
							tangent[i][j][k][l] =  input_tensor.index( { I, J }).item<double>();
						}}}}
		} else if (dim == 2) {
			// Mapping for 2D (assuming ordering: 0 -> (0,0), 1 -> (1,1), 2 -> (0,1))
			auto voigt_index = [](int i, int j) -> int {
				if (i > j)
					std::swap(i, j);
				if (i == 0 && j == 0)
					return 0;
				else if (i == 1 && j == 1)
					return 1;
				else if (i == 0 && j == 1)
					//					return 2;
					return 5;
				throw std::logic_error("Invalid indices for 2D Voigt mapping");
			};

			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					
					int I = voigt_index(i, j);
					stress[i][j] = stress_tensor.index( {0, I}).item<double>();
					
					for (int k = 0; k < 2; ++k) {
						for (int l = 0; l < 2; ++l) {
							int J = voigt_index(k, l);
							tangent[i][j][k][l] = input_tensor.index( { I, J }).item<double>();
						}}}}
		} else {
			throw std::logic_error("Dimension not supported!");
		}
}	


template <int dim>
void PlasticityProblem<dim>::get_stress_strain_tensor(
	const Eigen::Matrix<double, 6, 1> &B,
	const Eigen::Matrix<double, 6, 6> &A,
    SymmetricTensor<2, dim>       &stress,
    SymmetricTensor<4, dim>       &tangent)
{
		if (dim == 3) {
			// Define mapping for 3D (Voigt notation)
			auto voigt_index = [](int i, int j) -> int {
				if (i > j)
					std::swap(i, j);
				if (i == 0 && j == 0)
					return 0;
				else if (i == 1 && j == 1)
					return 1;
				else if (i == 2 && j == 2)
					return 2;
				else if (i == 1 && j == 2)
					return 3;
				else if (i == 0 && j == 2)
					return 4;
				else if (i == 0 && j == 1)
					return 5;
				throw std::logic_error("Invalid indices for Voigt mapping");
			};

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					
					int I = voigt_index(i, j);
					stress[i][j] = B(I);
					
					for (int k = 0; k < 3; ++k) {
						for (int l = 0; l < 3; ++l) {
							int J = voigt_index(k, l);
							tangent[i][j][k][l] = A(I, J);
						}}}}
		} else if (dim == 2) {
			// Mapping for 2D (assuming ordering: 0 -> (0,0), 1 -> (1,1), 2 -> (0,1))
			auto voigt_index = [](int i, int j) -> int {
				if (i > j)
					std::swap(i, j);
				if (i == 0 && j == 0)
					return 0;
				else if (i == 1 && j == 1)
					return 1;
				else if (i == 0 && j == 1)
//					return 2;
					return 5;
				throw std::logic_error("Invalid indices for 2D Voigt mapping");
			};

			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					
					int I = voigt_index(i, j);
					stress[i][j] = B(I);
					
					for (int k = 0; k < 2; ++k) {
						for (int l = 0; l < 2; ++l) {
							int J = voigt_index(k, l);
							tangent[i][j][k][l] = A(I, J);
						}}}}
		} else {
			throw std::logic_error("Dimension not supported!");
		}
}


template <int dim>
PlasticityProblem<dim>::PlasticityProblem()
  : mpi_communicator(MPI_COMM_WORLD), 
	n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)), 
	this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
	triangulation(mpi_communicator,
                typename Triangulation<dim>::MeshSmoothing(
                  Triangulation<dim>::smoothing_on_refinement |
                  Triangulation<dim>::smoothing_on_coarsening)),
	pcout(std::cout, (this_mpi_process == 0)), 
	computing_timer(mpi_communicator,
                  pcout,
                  TimerOutput::never,
                  TimerOutput::wall_times),
	
	fe(FE_Q<dim>(1) ^ dim),
    dof_handler(triangulation),
	quadrature_formula(fe.degree + 1),
    tolerance(1e-8),
    max_newton_iterations(50),
//	n_load_steps(20)
	n_load_steps(10)
{}


template<int dim>
void PlasticityProblem<dim>::setup_system() {
	TimerOutput::Scope t(computing_timer, "setup");
	dof_handler.distribute_dofs(fe);
	
	pcout << "   Number of active cells:       "
	            << triangulation.n_global_active_cells() << std::endl
	            << "   Number of degrees of freedom: " << dof_handler.n_dofs()
	            << std::endl;
	
	locally_owned_dofs = dof_handler.locally_owned_dofs();
	locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
	
	current_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
	newton_update.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
	incremental_displacement.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator); 
		
	system_rhs.reinit(locally_owned_dofs, mpi_communicator);

	zero_constraints.clear();
	zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
	DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
	VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(0), Functions::ZeroFunction <dim> (dim), zero_constraints);
	zero_constraints.close();

	DynamicSparsityPattern dsp(locally_relevant_dofs);
	DoFTools::make_sparsity_pattern(dof_handler, dsp, zero_constraints, /*keep_constrained_dofs = */ false);
	SparsityTools::distribute_sparsity_pattern(dsp,
											 dof_handler.locally_owned_dofs(),
											 mpi_communicator,
											 locally_relevant_dofs);
	system_matrix.reinit(locally_owned_dofs,
					   locally_owned_dofs,
					   dsp,
					   mpi_communicator);

	

	nonzero_constraints.clear();
	nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
	DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
	VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(0), Functions::ZeroFunction <dim> (dim), nonzero_constraints); // prescribed_displacement,                     
	nonzero_constraints.close();


	quadrature_point_data.resize(triangulation.n_active_cells());
//	tmp.resize(triangulation.n_active_cells());
	unsigned int cell_index = 0;
	for (auto &cell : triangulation.active_cell_iterators())
	{
		quadrature_point_data[cell_index].resize(quadrature_formula.size());
//		tmp[cell_index].resize(quadrature_formula.size());
		++cell_index;
	}
}


template<int dim>
void PlasticityProblem<dim>::assemble_system(const double current_load_factor, std::vector<std::vector<PointHistory<dim>>> &temp_ip_data) {
	TimerOutput::Scope t(computing_timer, "assembly");
	
	system_matrix = 0;
	system_rhs = 0;

	Tensor <1, dim> body_force;
	body_force[0] = current_load_factor * 10000.0;
	for (unsigned int d = 1; d < dim; ++d)
		body_force[d] = 0.0;

	//  RightHandSide<dim>          right_hand_side;
	//  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim));

	//  Tensor<1, dim> traction;
	//    traction[0] = 0.0;
	//    if (dim > 1)
	//      traction[1] = current_load_factor * 1e5; 
	//    else
	//      traction[0] = current_load_factor * 1e5;

	//  BoundaryForce<dim> boundary_force;
	//  std::vector<Vector<double>> boundary_force_values(n_face_q_points, Vector<double>(dim));

	
	FEValues <dim> fe_values(fe, quadrature_formula,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);

	const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
	FEFaceValues <dim> fe_face_values(fe, face_quadrature_formula,
			update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	const unsigned int n_q_points = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);
	//  std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);

	std::vector <types::global_dof_index> local_dof_indices(dofs_per_cell);

	const FEValuesExtractors::Vector displacement(0);
	
	
	unsigned int cell_counter = 0;
	for (auto &cell : dof_handler.active_cell_iterators()) {
		if (cell->is_locally_owned()) {
		fe_values.reinit(cell);
		        
		cell_matrix = 0;
		cell_rhs = 0;
		
		//    right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
		
		std::vector<SymmetricTensor<2, dim>> strain_tensor_previous(n_q_points);
		fe_values[displacement].get_function_symmetric_gradients(previous_solution, strain_tensor_previous);
	
		std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);
		fe_values[displacement].get_function_symmetric_gradients(current_solution, strain_tensor);
		

		double cell_volume = 0.0;
		double cell_trace_avg = 0.0;
		for (unsigned int q = 0; q < n_q_points; ++q)
		{
			cell_volume += fe_values.JxW(q);
			cell_trace_avg += trace(strain_tensor[q]) / static_cast<int>(dim) * fe_values.JxW(q);
		}
		cell_trace_avg /= cell_volume;
		
		
		std::vector<SymmetricTensor<2, dim>> strain_tensor_bar(n_q_points);
		for (unsigned int q = 0; q < n_q_points; ++q)
		{
		  strain_tensor_bar[q] = deviator(strain_tensor[q]) + cell_trace_avg * unit_symmetric_tensor<dim>();
		}
		
		
		
		cell_volume = 0.0;
		cell_trace_avg = 0.0;
		for (unsigned int q = 0; q < n_q_points; ++q)
		{
			cell_volume += fe_values.JxW(q);
			cell_trace_avg += trace(strain_tensor_previous[q]) / static_cast<int>(dim) * fe_values.JxW(q);
		}
		cell_trace_avg /= cell_volume;
		
		
		std::vector<SymmetricTensor<2, dim>> strain_tensor_previous_bar(n_q_points);
		for (unsigned int q = 0; q < n_q_points; ++q)
		{
		  strain_tensor_previous_bar[q] = deviator(strain_tensor_previous[q]) + cell_trace_avg * unit_symmetric_tensor<dim>();
		}
				
		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
			
			//////////////////////////
			//// Return Mapping /////
			////////////////////////
			
			PointHistory<dim> ip_data = temp_ip_data[cell_counter][q_point];
//			PointHistory<dim> &ip_data = quadrature_point_data[cell_counter][q_point];
			Vec6 dep;
//			dep << strain_tensor[q_point][0][0], strain_tensor[q_point][1][1], strain_tensor[q_point][2][2], strain_tensor[q_point][1][2], strain_tensor[q_point][0][2], strain_tensor[q_point][0][1];
//			dep << strain_tensor[q_point][0][0], strain_tensor[q_point][1][1], strain_tensor[q_point][0][0]*0.0, strain_tensor[q_point][0][0]*0.0, strain_tensor[q_point][0][0]*0.0, 2.0*strain_tensor[q_point][0][1];
			
			SymmetricTensor<2, dim> ep = strain_tensor_bar[q_point] - strain_tensor_previous_bar[q_point];
			dep << ep[0][0], ep[1][1], ep[0][0]*0.0, ep[0][0]*0.0, ep[0][0]*0.0, 2.0*ep[0][1];
			J2Material new_state = ip_data.material_point.returnMapping(dep, "plastic", 0.5, 1e9);
			
			 

			SymmetricTensor<2, dim> stress;
			SymmetricTensor<4, dim> tangent;
			get_stress_strain_tensor(new_state.sig, new_state.Ct, stress, tangent);
			
			temp_ip_data[cell_counter][q_point].material_point = new_state;	
			temp_ip_data[cell_counter][q_point].strain_xx = strain_tensor_bar[q_point][0][0];	
			temp_ip_data[cell_counter][q_point].strain_yy = strain_tensor_bar[q_point][1][1];	
			temp_ip_data[cell_counter][q_point].strain_xy = strain_tensor_bar[q_point][0][1];	
	
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				
				SymmetricTensor<2, dim> left;
				double cell_volume = 0.0;
				double cell_trace_avg = 0.0;
				for (unsigned int q = 0; q < n_q_points; ++q)
				{
					cell_volume += fe_values.JxW(q);
					cell_trace_avg += trace(fe_values[displacement].symmetric_gradient(i, q)) / static_cast<int>(dim) * fe_values.JxW(q);

				}
				cell_trace_avg /= cell_volume;
				left = deviator(fe_values[displacement].symmetric_gradient(i, q_point)) + cell_trace_avg * unit_symmetric_tensor<dim>();
				
								
//				const SymmetricTensor<2, dim> stress_phi_i = tangent*fe_values[displacement].symmetric_gradient(i,q_point);

				//        const unsigned int component_i = fe.system_to_component_index(i).first;
				//        cell_rhs(i) -= body_force_values[q_point](component_i) * fe_values.shape_value(i, q_point)* fe_values.JxW(q_point);
				
				cell_rhs(i) -= (left * stress * fe_values.JxW(q_point));	
//				cell_rhs(i) -= (fe_values[displacement].symmetric_gradient(i,q_point) * stress * fe_values.JxW(q_point));	
//				cell_rhs(i) -= (fe_values[displacement].symmetric_gradient(i,q_point) * tangent * strain_tensor[q_point] * fe_values.JxW(q_point));	
				cell_rhs(i) += fe_values[displacement].value(i, q_point) * body_force * fe_values.JxW(q_point);

				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
					
					SymmetricTensor<2, dim> right;
					double cell_volume = 0.0;
					double cell_trace_avg = 0.0;
					for (unsigned int q = 0; q < n_q_points; ++q)
					{
						cell_volume += fe_values.JxW(q);
						cell_trace_avg += trace(fe_values[displacement].symmetric_gradient(j, q)) / static_cast<int>(dim) * fe_values.JxW(q);

					}
					cell_trace_avg /= cell_volume;
					right = deviator(fe_values[displacement].symmetric_gradient(j, q_point)) + cell_trace_avg * unit_symmetric_tensor<dim>();

										
//					cell_matrix(i, j) += (fe_values[displacement].symmetric_gradient(i,q_point) * tangent * fe_values[displacement].symmetric_gradient(j,q_point) * fe_values.JxW(q_point));
					cell_matrix(i, j) += (left * tangent * right * fe_values.JxW(q_point));
				}
			}
		}
		
		           
		               
//		for (auto &face : cell->face_iterators()) {
//			if (face->at_boundary() && face->boundary_id() == 1) {
//				fe_face_values.reinit(cell, face);
//				//                      boundary_force.vector_value_list(
//				//                        fe_values_face.get_quadrature_points(),
//				//                        boundary_force_values);
//				for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
//					//    	                      Tensor<1, dim> traction;
//					//                              traction[2] = boundary_force_values[q_point][2];
//					for (unsigned int i = 0; i < dofs_per_cell; ++i) {
//						cell_rhs(i) += (fe_face_values[displacement].value(i,
//								q_point) * traction
//								* fe_face_values.JxW(q_point)); }
//				}
//			}
//		}
		cell->get_dof_indices(local_dof_indices);
		zero_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
				local_dof_indices, system_matrix, system_rhs); }
		
		++cell_counter;

	}
	
	system_matrix.compress(VectorOperation::add);
	system_rhs.compress(VectorOperation::add);
	       
}



template <int dim>
void PlasticityProblem<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "solve");
  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(system_rhs.size(), 1e-6 * system_rhs.l2_norm());   //dof_handler.n_dofs()
  LA::SolverCG  solver(solver_control);


  LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
  data.symmetric_operator = true;
#else
  /* Trilinos defaults are good */
#endif
  LA::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix, data);

  solver.solve(system_matrix,
			   completely_distributed_solution,
			   system_rhs,
			   preconditioner);

  pcout << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;

  zero_constraints.distribute(completely_distributed_solution);
  newton_update = completely_distributed_solution;
  
  current_solution.add(1.0, newton_update);
}
//template <int dim>
//void PlasticityProblem<dim>::refine_grid()
//{
//  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
//
//  KellyErrorEstimator<dim>::estimate(dof_handler,
//                                     QGauss<dim - 1>(fe.degree + 1),
//                                     {},
//                                     current_solution,
//                                     estimated_error_per_cell);
//
//  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
//                                                  estimated_error_per_cell,
//                                                  0.3,
//                                                  0.03);
//
//  triangulation.prepare_coarsening_and_refinement();
//
//  SolutionTransfer<dim> solution_transfer(dof_handler);
//  const Vector<double>  coarse_solution = current_solution;
//  solution_transfer.prepare_for_coarsening_and_refinement(coarse_solution);
//
//  triangulation.execute_coarsening_and_refinement();
//
//  setup_system();
//
//  solution_transfer.interpolate(coarse_solution, current_solution);
//
//  nonzero_constraints.distribute(current_solution);
//}


	
	


template <int dim>
void PlasticityProblem<dim>::output_results(const unsigned int load_step) const {
  TimerOutput::Scope t(computing_timer, "output");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  	std::vector <std::string> solution_names;
  	switch (dim) {
  	case 1:
  		solution_names.emplace_back("displacement");
  		break;
  	case 2:
  		solution_names.emplace_back("x_displacement");
  		solution_names.emplace_back("y_displacement");
  		break;
  	case 3:
  		solution_names.emplace_back("x_displacement");
  		solution_names.emplace_back("y_displacement");
  		solution_names.emplace_back("z_displacement");
  		break;
  //	default:
  //		DEAL_II_NOT_IMPLEMENTED();
  	}

  	data_out.add_data_vector(current_solution, solution_names);
	/////////////////
//  		std::ofstream output_file("cv/gauss_points" + std::to_string(load_step) + ".csv");
//  			output_file << "x,y,epsilon_xx,epsilon_yy,epsilon_xy\n";
//
//  			FEValues <dim> fe_values(fe, quadrature_formula,
//  						update_values | update_gradients | update_quadrature_points | update_JxW_values);
//  			    
//  			unsigned int cell_counter = 0;
//  					for (auto &cell : triangulation.active_cell_iterators()) {
//  						fe_values.reinit(cell);
//  							
//  				for (unsigned int q_point = 0; q_point < quadrature_formula.size(); ++q_point) {
//  					
//  					
//  					const Point<dim> &p = fe_values.get_quadrature_points()[q_point]; 
//
//
//  					output_file << p(0) << "," << p(1) << ","
//  			             << quadrature_point_data[cell_counter][q_point].strain_xx << "," 
//  						 << quadrature_point_data[cell_counter][q_point].strain_yy << "," 
//  						 << quadrature_point_data[cell_counter][q_point].strain_xy << "\n";
//  				}
//  				++cell_counter;
//  			}
//  		    output_file.close();
	//////////////////
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
	subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();
  
  data_out.write_vtu_with_pvtu_record("./", "solution", load_step, mpi_communicator, 2, 8);
}
      
template<int dim>
void PlasticityProblem<dim>::run() {
	
	pcout << "Running with "
	#ifdef USE_PETSC_LA
			<< "PETSc"
	#else
			<< "Trilinos"
	#endif
			<< " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
			<< " MPI rank(s)..." << std::endl;
	
	GridGenerator::hyper_cube(triangulation, 0, 1);

//Triangulation<2> tria_plate_hole;
//GridGenerator::hyper_cube_with_cylindrical_hole(tria_plate_hole,0.03,0.06);
//tria_plate_hole.refine_global(1);
//  
//std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;
//for (const auto &cell : tria_plate_hole.active_cell_iterators()) {
//	if (cell->center()[0] < 0.0 || cell->center()[1] < 0.0)
//		cells_to_remove.insert(cell); }
//GridGenerator::create_triangulation_with_removed_cells(tria_plate_hole,cells_to_remove,triangulation);	
//triangulation.refine_global(3);   
			
//	for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
//	          if (cell->vertex(v).square() < .1)
//	            cells_to_remove.insert(cell);
	
	triangulation.refine_global(4);
//	std::ofstream out("grid.svg");
//	GridOut       grid_out;
//	grid_out.write_svg(triangulation, out);
//	std::cout << "Grid written to grid.svg" << std::endl;


	setup_system();
	nonzero_constraints.distribute(current_solution);

	for (unsigned int load_step = 0; load_step < n_load_steps; ++load_step) {
		// Incremental stress update
		const double current_load_factor = (load_step + 1) / static_cast<double>(n_load_steps);
		std::cout << "\n=== Load step " << load_step + 1
				<< " with load factor = " << current_load_factor << " ==="
				<< std::endl;

		// Newton's update
		previous_solution = current_solution;
	
		for (unsigned int newton_step = 0; newton_step < max_newton_iterations; ++newton_step) {		
			
			std::vector<std::vector<PointHistory<dim>>> temp_ip_data = quadrature_point_data;
			
			
			assemble_system(current_load_factor, temp_ip_data); 
			solve();

			const double residual_norm = system_rhs.l2_norm();
			std::cout << "Newton iteration "
					<< newton_step + 1 << ": Residual norm = " << residual_norm
					<< std::endl;

			if (residual_norm < tolerance) {
				quadrature_point_data = temp_ip_data;
//				for (unsigned int cell=0; cell<quadrature_point_data.size(); ++cell)
//				          for (unsigned int q=0; q<quadrature_point_data[cell].size(); ++q)
//				            quadrature_point_data[cell][q] = tmp[cell][q];
				break; 
			}
		}
		
//		unsigned int cell_counter = 0;
//		for (auto &cell : dof_handler.active_cell_iterators()) {
//			for (unsigned int q_point = 0; q_point < quadrature_formula.size(); ++q_point) {	
//				quadrature_point_data[cell_counter][q_point].material_point = tmp[cell_counter][q_point].material_point;
//				++cell_counter;
//			}
//		}
//		quadrature_point_data.swap(tmp);

//		output_results(load_step + 1);
		computing_timer.print_summary();
		computing_timer.reset();
		pcout << std::endl;
	}
	pcout << current_solution << std::endl;

	
}

int main(int argc, char *argv[]) {

	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	
	PlasticityProblem<2> problem;
	problem.run();

}


