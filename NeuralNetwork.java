/*
 * Two layer neural network
 * Three hidden neurons
 * Binary classification
 * Rectified linear unit non-linearity on each neuron
 * a*x + b*y + c*z + d
 */
import java.util.*;
public class NeuralNetwork {
	public static void main(String[] args) {
		// Each data point act as an input
		// Three features per data point
		// Expected output of -1 or 1
		Feature f1 = new Feature(255, 0, 0);
		Feature f2 = new Feature(243, 88, 88);
		Feature f3 = new Feature(183, 25, 25);
		Feature f4 = new Feature(25, 52, 183);
		Feature f5 = new Feature(10, 49, 247);
		Feature f6 = new Feature(0, 0, 255);
		Feature[] data= {f1, f2, f3, f4, f5, f6};
		double[] expected_output = {1, -1, 1, -1, -1, 1};
		// Size acts as amount of neurons + 1
		// Weight shows starting point of each weight in array
		int size = 4;
		int weight1 = size*0;
		int weight2 = size*1;
		int weight3 = size*2;
		int weight4 = size*3;
		double forward[] = new double[size*4 + 1];
		double back[] = new double[size*4 + 1];
		// Initial values for forward pass as random value between -0.5 and 0.5
		// Initial values for back propagation as 0
		for(double var: forward) {
			var = Math.random() - 0.5;
		}
		for(double var2: back) {
			var2 = 0.0;
		}
		// Loop to train values through iterations
		for(int iter = 0; iter <= 1000; iter++) {
			// Use random data point
			int i = (int)Math.floor(Math.random() * data.length);
			// Find feature values and expected output for data point
			double x = data[i].first;
			double y = data[i].second;
			double z = data[i].third;
			double output = expected_output[i];
			// Activate neurons using a*x + b*y + c*z + d
			double neuron1 = Math.max(0, forward[weight1]*x + forward[weight2]*y + forward[weight3]*z + forward[weight4]);
			double neuron2 = Math.max(0, forward[weight1 + 1]*x + forward[weight2 + 1]*y + forward[weight3 + 1]*z + forward[weight4 + 1]);
			double neuron3 = Math.max(0, forward[weight1 + 2]*x + forward[weight2 + 2]*y + forward[weight3 + 2]*z + forward[weight4 + 2]);
			// Find final score using three neurons a*n1 + b*n2 + c*n3 + d;
			double score = forward[weight1 + 3]*neuron1 + forward[weight2 + 3]*neuron2 + forward[weight3 + 3]*neuron3 + forward[forward.length-1];//?? 
			// Print accuracy every 50 iterations
			if(iter % 50 == 0) {
				System.out.println("Iteration: " + iter + "\t Training accuracy: " + eval(data, expected_output, forward));
				if(eval(data, expected_output, forward) == 1.0) {
					break;
				}
			}
			double pull = 0.0;
			// If output and score trend doesn't match
			// Pull higher or lower accordingly
			if(output == 1 && score < 1) {
				pull = 1;
			}
			if(output == -1 && score > -1) {
				pull = -1;
			}
			// Back propagate to every parameter in the model
			// Back prop through last score neuron
			double dscore = pull;		
			back[weight1 + 3] = neuron1 * dscore;
			back[weight2 + 3] = neuron2 * dscore;
			back[weight3 + 3] = neuron3 * dscore;
			back[forward.length-1] = 1.0 * dscore; //??
			double dneuron1 = forward[weight1 + 3] * dscore;
			double dneuron2 = forward[weight2 + 3] * dscore;
			double dneuron3 = forward[weight3 + 3] * dscore;

			// Back prop non linearities or set gradients to 0 if not fired
			dneuron3 = neuron3 == 0 ? 0 : dneuron3;
			dneuron2 = neuron2 == 0 ? 0 : dneuron2;
			dneuron1 = neuron1 == 0 ? 0 : dneuron1;
			
			// Back prop to parameters of neuron1
			back[weight1 + 0] = x * dneuron1;
			back[weight2 + 0] = y * dneuron1;
			back[weight3 + 0] = z * dneuron1;
			back[weight4 + 0] = 1.0 * dneuron1;
			
			// Back prop to parameters of neuron2
			back[weight1 + 1] = x * dneuron2;
			back[weight2 + 1] = y * dneuron2;
			back[weight3 + 1] = z * dneuron2;
			back[weight4 + 1] = 1.0 * dneuron2;

			// Back prop to parameters of neuron3
			back[weight1 + 2] = x * dneuron3;
			back[weight2 + 2] = y * dneuron3;
			back[weight3 + 2] = z * dneuron3;
			back[weight4 + 2] = 1.0 * dneuron3;
			
			// Add pulls from the regularization
			// Tug all multiplicative parameters downward proportional to value
			for(int j = 0; j < size*3; j++) {
				if((j+1)%4 != 0) {
					back[j] += -forward[j];
				}
			}
			
			// Update parameters
		    double step_size = 0.01;
			for(int j = 0; j < forward.length; j++) {
				forward[j] += step_size * back[j];
			}
		}
		Scanner sc = new Scanner(System.in);
		System.out.print("R: ");
		double red = (double)sc.nextInt();
		System.out.print("G: ");
		double green = (double)sc.nextInt();
		System.out.print("B: ");
		double blue = (double)sc.nextInt();
		double temp = calc(forward, red, green, blue);
		if(temp > 0) {
			System.out.println("Red detected");
		} else {
			System.out.println("Blue detected");
		}
	}
	// Evaluate accuracy of training at given iteration
	public static double eval(Feature[] data, double[] expected_output, double[] forward) {
		double num_correct = 0;
		for(int i = 0; i < data.length; i++) {
			double x = data[i].first;
			double y = data[i].second;
			double z = data[i].third;
			double true_output = expected_output[i];
			double predicted_output = (calc(forward, x, y, z)) > 0 ? 1 : -1;
			// Compare predicted output to expected output
			if(predicted_output == true_output) {
				num_correct++;
		    }
		}
		return num_correct/data.length;
	}
	// Calculate predicted output
	public static double calc(double[] forward, double x, double y, double z) {
		int size = 4;
		int weight1 = size*0;
		int weight2 = size*1;
		int weight3 = size*2;
		int weight4 = size*3;
		// Calculate neurons with a*x + b*y + c*z + d
		double neuron1 = Math.max(0, forward[weight1]*x + forward[weight2]*y + forward[weight3]*z + forward[weight4]);
		double neuron2 = Math.max(0, forward[weight1 + 1]*x + forward[weight2 + 1]*y + forward[weight3 + 1]*z + forward[weight4 + 1]);
		double neuron3 = Math.max(0, forward[weight1 + 2]*x + forward[weight2 + 2]*y + forward[weight3 + 2]*z + forward[weight4 + 2]);
		// Find final score using three neurons a*n1 + b*n2 + c*n3 + d;
		return forward[weight1 + 3]*neuron1 + forward[weight2 + 3]*neuron2 + forward[weight3 + 3]*neuron3 + forward[forward.length-1]; 
	}
}

/* 
 * Feature class holds each data points
 * Has three features with double as data type
 */
class Feature {
	double first;
	double second;
	double third;
	public Feature(double one, double two, double three) {
		this.first = one;
		this.second = two;
		this.third = three;
	}
}