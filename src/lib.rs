pub mod node2vec {
    use std::error::Error;
    use nalgebra::DMatrix;
    use petgraph::graph::{Graph, NodeIndex};
    use petgraph::{EdgeType, Directed};
    use petgraph::visit::EdgeRef;
    use rand::{thread_rng, SeedableRng, Rng};
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use std::collections::HashMap;

    const NEG_POW: f32 = 0.75;
    const MAX_SIGMOID: f32 = 8.0;
    const SIGMOID_TABLE_SIZE: usize = 512;
    const NEGATIVE_TABLE_SIZE: usize = 100000;
    const LOG_TABLE_SIZE: usize = 512;

    fn get_neighbor_info(graph: &Graph<usize, f32, Directed>) -> HashMap<NodeIndex, Vec<(NodeIndex, f32)>> {
        let mut neighbor_info: HashMap<NodeIndex, Vec<(NodeIndex, f32)>> = HashMap::new();
        for edge in graph.edge_references() {
            let source = edge.source();
            let target = edge.target();
            let weight = *edge.weight();
            if neighbor_info.contains_key(&source) {
                neighbor_info.get_mut(&source).unwrap().push((target, weight));
            } else {
                neighbor_info.insert(source, vec![(target, weight)]);
            }
        }

        for (_node, neighbors) in neighbor_info.iter_mut() {
            let mut neighbor_weight_sum: f32 = 0.0;
            for (_neighbor, weight) in neighbors.clone() {
                neighbor_weight_sum += weight;
            }
            for (_neighbor, weight) in neighbors {
                *weight /= neighbor_weight_sum;
            }
        }

        neighbor_info
    }

    fn random_walk(graph: &Graph<usize, f32, Directed>, start_node: NodeIndex, walk_length: usize) -> Vec<NodeIndex> {
        let neighbor_info = get_neighbor_info(graph);
        let mut walk: Vec<NodeIndex> = Vec::new();
        let mut current_node = start_node;
        for _ in 0..walk_length {
            if walk.contains(&current_node) {
                break;
            } else {
                walk.push(current_node);
    
                let neighbors = neighbor_info.get(&current_node).unwrap();
                let mut probability_table: Vec<f32> = Vec::new();
                for (_i, weight) in neighbors {
                    if probability_table.is_empty() {
                        probability_table.push(*weight);
                    } else {
                        probability_table.push(probability_table.last().unwrap() + *weight);
                    }
                }

                let mut rng = rand::thread_rng();
                let random_number = rng.gen_range(0.0..1.0) as f32;
                for (i, prob) in probability_table.iter().enumerate() {
                    if random_number < *prob {
                        current_node = neighbors[i].0;
                        break;
                    }
                }
            }
        }
    
        walk
    }

    fn normalize_graph<Ty: EdgeType>(graph: &petgraph::Graph<usize, f32, Ty>) -> Graph<usize, f32, Directed> {
        let mut normalized_graph: Graph<usize, f32, Directed> = Graph::new();
        if Ty::is_directed() {
            for node in graph.node_indices() {
                normalized_graph.add_node(node.index());
            }
    
            for edge in graph.edge_references() {
                let source = edge.source();
                let target = edge.target();
                let weight = *edge.weight();
                normalized_graph.add_edge(source, target, weight);
            }
        } else {
            for node in graph.node_indices() {
                normalized_graph.add_node(node.index());
            }
        
            for edge in graph.edge_references() {
                let source = edge.source();
                let target = edge.target();
                let weight = *edge.weight();
                normalized_graph.add_edge(source, target, weight);
                normalized_graph.add_edge(target, source, weight);
            }
        };

        normalized_graph
    }
    
    fn sample<Ty: EdgeType>(graph: &petgraph::Graph<usize, f32, Ty>, walk_length: usize, num_walks: usize) -> Vec<Vec<NodeIndex>> {
        let normalized_graph: Graph<usize, f32, Directed> = normalize_graph(graph);

        let mut walks: Vec<Vec<NodeIndex>> = Vec::new();
        for _ in 0..num_walks {
            for node in normalized_graph.node_indices() {
                let walk = random_walk(&normalized_graph.clone(), node, walk_length);
                walks.push(walk);
            }
        }
    
        walks
    }

    struct TrainArgument {
        input: Vec<Vec<NodeIndex>>,
        embed_dim: usize,
        lr: f32,
        win: u32,
        epoch: u32,
        neg: usize,
        threshold: f32,
        lr_update: u32,
    }

    #[allow(dead_code)]
    pub struct Node2vec {
        pub node_embedding: DMatrix<f32>,
        dict: Dict,
    }

    struct Dict {
        node2ent: HashMap<NodeIndex, Entry>,
        ntokens: usize,
    }

    #[derive(Clone, Debug)]
    struct Entry {
        index: usize,
        count: u32,
    }

    impl Dict {
        fn form_dict(sequences: Vec<Vec<NodeIndex>>) -> Dict {
            let mut new_node2ent: HashMap<NodeIndex, Entry> = HashMap::new();
            let mut new_idx2node: HashMap<usize, NodeIndex> = HashMap::new();
            let mut new_ntokens: usize = 0;
            for sequence in sequences.clone() {
                for token in sequence {
                    if new_node2ent.contains_key(&token) {
                        let new_count = new_node2ent[&token].count + 1;
                        let new_entry = Entry {
                            index: new_node2ent[&token].index,
                            count: new_count,
                        };
                        new_node2ent.insert(token.clone(), new_entry);
                    } else {
                        let ent = Entry {
                            index: new_ntokens,
                            count: 1,
                        };
                        new_node2ent.insert(token.clone(), ent);
                        new_idx2node.insert(new_ntokens, token.clone());
                        new_ntokens += 1;
                    }
                }
            }
    
            Dict {
                node2ent: new_node2ent,
                ntokens: new_ntokens,
            }
        }
    
        fn nsize(&self) -> usize {
    
            self.ntokens
        }
    
        fn get_idx(&self, node: &NodeIndex) -> usize {
    
            self.node2ent[node].index
        }
    
        // fn get_node(&self, idx: usize) -> NodeIndex {
    
        //     self.idx2node[&idx].clone()
        // }
    
        // fn get_entry(&self, node: &NodeIndex) -> Entry {
    
        //     Entry {
        //         index: self.node2ent[node].index,
        //         count: self.node2ent[node].count,
        //     }
        // }
    
        fn get_counts(&self) -> Vec<u32> {
            let mut counts: Vec<u32> = vec![0; self.ntokens];
            for (_node, ent) in &self.node2ent {
                counts[ent.index] = ent.count;
            }
    
            counts
        }
    
        fn init_negative_table(&self) -> Vec<usize> {
            let mut negative_table: Vec<usize> = Vec::new();
            let counts_vec = self.get_counts();
    
            let mut z: f32 = 0.0;
            for c in counts_vec.clone() {
                z += (c as f32).powf(NEG_POW);
            }
    
            for (idx, i) in counts_vec.into_iter().enumerate() {
                let c = (i as f32).powf(NEG_POW);
                for _ in 0..(c * (NEGATIVE_TABLE_SIZE as f32) / z) as usize {
                    negative_table.push(idx as usize);
                }
            }
            let mut rng = thread_rng();
            negative_table.shuffle(&mut rng);
            
            negative_table
        }
    }

    struct ModelTmpStat {
        node_embeddings: DMatrix<f32>,
        embed_dim: usize,
        lr: f32,
        neg: usize,
        grad: Vec<f32>,
        neg_pos: usize,
        sigmoid_table: Vec<f32>,
        log_table: Vec<f32>,
        negative_table: Vec<usize>,
        loss: f32,
        nsamples: u32,
    }

    impl ModelTmpStat {
        fn new(
            node_embeddings: DMatrix<f32>,
            embed_dim: usize,
            lr: f32,
            neg: usize,
            neg_table: Vec<usize>,
        ) -> ModelTmpStat {

            let mut grad: Vec<f32> = Vec::new();
            for _ in 0..embed_dim {
                grad.push(0.0);
            }
    
            ModelTmpStat {
                node_embeddings: node_embeddings,
                embed_dim: embed_dim,
                lr: lr,
                neg: neg,
                grad: grad,
                neg_pos: 0,
                sigmoid_table: init_sigmoid_table(),
                log_table: init_log_table(),
                negative_table: neg_table,
                loss: 0.,
                nsamples: 0,
            }
        }
    
        fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
    
        fn update(&mut self, token_id: usize, target: usize) {
            self.loss += self.sampling(token_id, target);
            self.nsamples += 1;
        }
    
        fn sampling(&mut self, token_id: usize, target_index: usize) -> f32 {
            let mut loss: f32 = 0.0;
            for i in 0..(self.neg + 1) {
                if i == 0 {
                    loss += self.binary_losgistic(token_id, target_index, 1);
                } else {
                    let neg_sample = self.get_negative(target_index);
                    loss += self.binary_losgistic(token_id, neg_sample, 0);
                }
            }
    
            let grad_to_add = DMatrix::from_vec(1, self.grad.len(), self.grad.clone());
            for i in 0..self.embed_dim {
                self.node_embeddings[(token_id, i)] += grad_to_add[(0, i)];
            }
            self.grad = vec![0.0f32; self.embed_dim];
    
            loss
        }
    
        fn binary_losgistic(&mut self, input_id: usize, target_id: usize, label: i32) -> f32 {
            let input_embedding = self.node_embeddings.row(input_id);
            let target_embedding = self.node_embeddings.row(target_id);
            let sum = input_embedding.dot(&target_embedding);
            let score = self.sigmoid(sum as f32);
            let alpha = self.lr * (label as f32 - score);
            
            let add_to_grad: Vec<f32> = (target_embedding * alpha).row(0).iter().cloned().collect();
            for i in 0..self.embed_dim {
                self.grad[i] += add_to_grad[i];
            }
            let add_to_model_output = input_embedding * alpha;
            for i in 0..self.embed_dim {
                self.node_embeddings[(target_id, i)] += add_to_model_output[i];
            }
            
            if label == 1 {
                -self.log(score)
            } else {
                -self.log(1.0 - score)
            }
        }
    
        fn sigmoid(&self, x: f32) -> f32 {
            let sigmoid_result: f32 = if x < -MAX_SIGMOID {
                0.0
            } else if x > MAX_SIGMOID {
                1.0
            } else {
                let i = (((x + MAX_SIGMOID) * (SIGMOID_TABLE_SIZE as f32)) / MAX_SIGMOID) / 2.0;
                self.sigmoid_table[i as usize]
            };
    
            sigmoid_result
        }
    
        fn log(&self, x: f32) -> f32 {
            let result = if x > 1.0 {
                x
            } else {
                let i = (x * (LOG_TABLE_SIZE as f32)) as usize;
                self.log_table[i]
            };
    
            result
        }
        
        fn get_negative(&mut self, target: usize) -> usize {
            loop {
                let negative = self.negative_table[self.neg_pos];
                self.neg_pos = (self.neg_pos + 1) % self.negative_table.len();
                if target != negative {
                    break negative;
                }
            }
        }
    }

    fn init_sigmoid_table() -> Vec<f32> {
        let mut sigmoid_table: Vec<f32> = vec![0.0; SIGMOID_TABLE_SIZE + 1];
        for i in 0..(SIGMOID_TABLE_SIZE + 1) {
            let x = (((i as f32) * 2.0 * MAX_SIGMOID) / (SIGMOID_TABLE_SIZE as f32)) - MAX_SIGMOID;
            sigmoid_table[i] = 1.0 / (1.0 + (-x).exp());
        }
        
        sigmoid_table
    }
    
    fn init_log_table() -> Vec<f32> {
        let mut log_table: Vec<f32> = vec![0.0; LOG_TABLE_SIZE + 1];
        for i in 0..(LOG_TABLE_SIZE + 1) {
            let x = (i as f32 + 1e-5) / (LOG_TABLE_SIZE as f32);
            log_table[i] = x.ln();
        }
    
        log_table
    }

    fn skipgram(model: &mut ModelTmpStat, token_id: usize, bound: i32) {
        let length = model.node_embeddings.nrows();
        for c in (-bound)..(bound + 1) {
            if c != 0 && ((token_id as i32) + c) >= 0 && ((token_id as i32) + c) < (length as i32) {
                model.update(token_id, ((token_id as i32) + c) as usize);
            }
        }
    }

    fn train(args: &TrainArgument) -> Result<Node2vec, Box<dyn Error>> {
        let dict = Dict::form_dict(args.input.clone());
        let mut input_mat = DMatrix::<f32>::zeros(dict.nsize(), args.embed_dim);
        let seed_value = 42;
        let mut rng = StdRng::seed_from_u64(seed_value);
        for i in 0..input_mat.nrows() {
            for j in 0..input_mat.ncols() {
                input_mat[(i, j)] = rng.gen_range((-1.0f32 / args.embed_dim as f32)..(1.0f32 / args.embed_dim as f32));
            }
        }
    
        let neg_table = dict.init_negative_table();
        let mut model = ModelTmpStat::new(
            input_mat.clone(),
            args.embed_dim,
            args.lr,
            args.neg,
            neg_table,
        );
        let mut token_count: u32 = 0;
        let all_tokens = args.epoch as usize * dict.ntokens;
    
        let mut tmp_node_embedding_state: DMatrix<f32> = DMatrix::zeros(input_mat.nrows(), args.embed_dim);
        let mut tmp_loss_state: f32 = f32::INFINITY;

        for _ in 0..args.epoch {
            for sequence in args.input.clone() {
                for seq in sequence {
                    let token_id = dict.get_idx(&seq);
                    skipgram(&mut model, token_id, args.win as i32);
                    if token_count > args.lr_update as u32 {
                        let count = token_count as f32;
                        let progress = count / all_tokens as f32;
                        model.set_lr(args.lr * (1.0 - progress));
                        token_count = 0;
                    }
                }
            }
    
            if model.loss / model.nsamples as f32 > tmp_loss_state - args.threshold {
                tmp_node_embedding_state = model.node_embeddings.clone();
                break;
            } else {
                tmp_loss_state = model.loss / model.nsamples as f32;
                tmp_node_embedding_state = model.node_embeddings.clone();
                model.loss = 0.0;
            }
        }
    
        let n2v = Node2vec {
            node_embedding: tmp_node_embedding_state,
            dict: dict,
        };
    
        Ok(n2v)
    }

    fn get_adequate_embedding_dimension(graph: &Graph<usize, f32, Directed>) -> usize {
        let tmp = (graph.node_count() as f32).log10().floor() as usize;
        
        if tmp < 2 {
            2
        } else {
            tmp
        }
    }

    pub fn embed<Ty: EdgeType>(graph: &petgraph::Graph<usize, f32, Ty>, walk_length: usize, num_walks: usize, result_embed_dimension: Option<usize>) -> Result<HashMap<NodeIndex, Vec<f32>>, Box<dyn Error>> {
        let walks = sample(graph, walk_length, num_walks);
        let dim = match result_embed_dimension {
            Some(d) => d,
            None => {
                let normalized_graph = normalize_graph(graph);
                get_adequate_embedding_dimension(&normalized_graph)
            }
        };

        let args = TrainArgument {
            input: walks,
            embed_dim: dim,
            lr: 0.001,
            win: 5,
            epoch: 100,
            neg: 3,
            threshold: 5e-2,
            lr_update: 10000,
        };

        let result = train(&args);

        match result {
            Ok(n2v) => {
                let mut node_embedding: HashMap<NodeIndex, Vec<f32>> = HashMap::new();
                for node in graph.node_indices() {
                    let idx = n2v.dict.get_idx(&node);
                    let mut embedding: Vec<f32> = Vec::new();
                    for i in 0..dim {
                        embedding.push(n2v.node_embedding[(idx, i)]);
                    }
                    node_embedding.insert(node, embedding);
                }

                Ok(node_embedding)
            },
            Err(e) => Err(e),
        }        
    }
}