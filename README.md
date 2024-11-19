# Node2Vec

This is a node2vec implementation in Rust. Node2Vec is an algorithmic framework for learning continuous feature representations for nodes in networks. The implementation is based on the paper [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) by Aditya Grover and Jure Leskovec.

However, this is not a simple application of the method suggested in the original paper, the changes are as follows: The original paper uses a biased random walk to generate the sequences. However, due to inefficient memory consumption and time complexity, there are some modification in the random walk generation process. The modified algorithm was based on following repository, ([GitHub Repo](https://github.com/louisabraham/fastnode2vec)), but I modified it further, to fit better in weighted graphs.

## Features

- **Automatic dimension adjustment**: The crate automatically adjusts the dimension of the feature vector according to the number of nodes in the graph.

- **Compatible with petgraph**: The crate does not use self-defined data type. Instead, it use petgraph crate, in order to improve the compatibility.

## Installation

To use this crate, add it to your `Cargo.toml` dependencies:

```
[dependencies]
node2vec = "1.0.1"
```

## Usage

```
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use nalgebra::DMatrix;
use petgraph::graph::Graph;
use node2vec::node2vec::embed;

fn main() {
    let mut graph: Graph<usize, f32, petgraph::Directed> = Graph::new(); // The petgraph type must be defined like this, with the definition of edge direction.
    let node1 = graph.add_node(1);
    let node2 = graph.add_node(2);
    let node3 = graph.add_node(3);
    let node4 = graph.add_node(4);
    let node5 = graph.add_node(5);
    let node6 = graph.add_node(6);
    let node7 = graph.add_node(7);
    let node8 = graph.add_node(8);
    let node9 = graph.add_node(9);
    let node10 = graph.add_node(10);
    graph.add_edge(node1, node2, 1.0);
    graph.add_edge(node1, node3, 1.0);
    graph.add_edge(node1, node4, 1.0);
    graph.add_edge(node2, node5, 1.0);
    graph.add_edge(node2, node6, 1.0);
    graph.add_edge(node3, node7, 1.0);
    graph.add_edge(node3, node8, 1.0);
    graph.add_edge(node4, node9, 1.0);
    graph.add_edge(node4, node10, 1.0);
    graph.add_edge(node5, node6, 1.0);
    graph.add_edge(node7, node8, 1.0);
    graph.add_edge(node9, node10, 1.0);
    
    let embed = embed(
        &graph, 
        20, 
        200, 
        None // None for automatic setting, Some(usize) for manual setting
    );

    println!("{:?}", embed);
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.