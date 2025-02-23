"""
Coarsening
----------
Description:
    A module containing different graph coarsening algorithms
"""
import networkx as nx 
from copy import deepcopy
import scipy 
import numpy as np
from sklearn.preprocessing import normalize

class CoarseNET:
    '''
    A class that implements the CoarseNET algorithm for an unweighted, 
    undirected graph.

    Attributes
    ----------
    G_coarse : networkx.Graph
        Coarsened graph.
    
    alpha : float
        Shrinkage factor
    '''
    
    def __init__(self,alpha: float,G: nx.Graph):
        '''
        Parameters
        ----------
        alpha : float
            Shrinkage factor.

        G : networkx.Graph
            Graph to coarsen.
        '''
        self.alpha = alpha
        self.G = deepcopy(G)
    
    @property
    def alpha(self):
        '''Shrinkage factor'''
        return self._alpha
    
    @alpha.setter
    def alpha(self,val):
        #TODO: Validate within range (0,1.0)
        self._alpha = val
        
    @property
    def G(self):
        '''Original Graph

        Returns
        -------
        G : networkx.Graph
            The original graph to miniaturize
        '''
        return self._G
    
    @G.setter
    def G(self,Graph: nx.Graph):
        '''
        Test
        Parameters
        ----------
        Graph : networkx.Graph
        '''
        #TODO: Validate strongly connected graph
        self._G = Graph
    
    @staticmethod
    def adjacency(G: nx.Graph):
        '''Returns the column-normalized adjacency matrix of
        a graph.

        Parameters
        ----------
        G : networkx.Graph
            A graph to construct the adjacency matrix.
        '''
        A = nx.to_scipy_sparse_array(G)
        A = A._asfptype()
        A = normalize(A,norm='l2',axis=0)
        
        return A
    
    @staticmethod
    def eigs(G: nx.Graph):
        '''Computes the dominant eigenvalue and eigenvectors
        associated with the adjacency matrix of a graph.

        Parameters
        ----------
        G : networkx.Graph
            Graph to calculate eigenvalue and eigenvectors.
        '''
        # Adjacency Matrix
        A = CoarseNET.adjacency(G)
        
        # Compute the first eigenvalue and right eigenvector
        lambda_, u_ = scipy.sparse.linalg.eigs(A,k=1)
        
        # Compute the left eigenvector
        _, v_= scipy.sparse.linalg.eigs(A.T,k=1)
                
        return np.real(lambda_)[0], np.real(np.squeeze(u_)), np.real(np.squeeze(v_))
    
    def __edge_score(self,edge):
        '''Calculates the score of a node pair
        '''
        u_a, u_b = self.u_[edge[0]], self.u_[edge[1]]
        v_a, v_b = self.v_[edge[0]], self.v_[edge[1]]
        
        prod = (self.lambda_-1)*(u_a+u_b)
        score = (-self.lambda_*(u_a*v_a+u_b*v_b) + v_a*prod + u_a*v_b + u_b*v_a) / (np.dot(self.v_,self.u_)-(u_a*v_a + u_b*v_b))  
    
        return score
        
    def __score(self):
        '''Calculates the score for all the edges in the graph
        '''
        # Initialize array of scores
        score = np.zeros(self.G_coarse_.number_of_edges())
        
        # Calculate score for every edge in the graph
        for i, edge in enumerate(self.G_coarse_.edges):
            score[i] = self.__edge_score(edge)
        
        return np.abs(score)
    
    def __contract(self,edge) -> bool:
        '''Updates graph by contracting nodes in the edge
        '''
        # Upack nodes
        u, v = edge
        left, right = self.nodes_coarse_[u], self.nodes_coarse_[v]
        
        contract = left != right
        if contract:
            # Merge nodes
            nx.contracted_nodes(self.G_coarse_,
                                left,
                                right,
                                self_loops=False,
                                copy=False)
            
            # Update node index in coarsened graph
            idx = self.nodes_coarse_ == right
            self.nodes_coarse_[idx] = left 
            self.nodes_removed_.append(right)
            
        return contract
        
    def coarsen(self) -> None:
        '''
        Coarsens the seed graph
        '''
        self.G_coarse_ = self._G.to_directed()
        n = self.G_coarse_.number_of_nodes()
        n_edges = self.G_coarse_.number_of_edges()
        n_reduced = int(self._alpha * n)
        
        # Compute the eigenvalue and eigenvectors
        self.lambda_, self.u_, self.v_ = CoarseNET.eigs(self.G_coarse_)
        
        # Arrays of nodes and edges
        self.nodes_coarse_ = np.arange(0,n,dtype=np.int32)
        self.nodes_removed_ = []
        edges = list(self.G_coarse_.edges)
        
        # Calculate sorting indices according to score
        score = self.__score()
        idx = np.argsort(score)
        
        contractions = 0
        i = 0
        while (contractions < n_reduced) and (i < n_edges):
            # Retrieve edge according to sorting
            edge = edges[idx[i]]
            
            # Contract edges
            contract = self.__contract(edge)
            
            if contract:
                contractions += 1
            
            i += 1

        # Add removed edges to the original graph
        #self.G_coarse_.add_nodes_from(self.nodes_removed_)