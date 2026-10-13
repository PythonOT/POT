/* -*- mode: C++; indent-tabs-mode: nil; -*-
 *
 * General (non-bipartite) sparse directed graph for optimal transport.
 *
 * Unlike SparseBipartiteDigraph (see sparse_bipartitegraph.h), nodes are not
 * split into a source half and a target half: any node may carry supply or
 * demand. This is required for min-cost-flow formulations on grid graphs
 * (e.g. EMD-L1 on Cartesian grids), where every cell is both a potential
 * source and a potential sink.
 *
 * Uses CSR (Compressed Sparse Row) format for cache-friendly arc iteration.
 * Requires edges to be provided in sorted order during construction (sorting
 * is done internally by buildFromEdges).
 */

#pragma once

#include "core.h"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <tuple>
#include <utility>

namespace lemon {

  class SparseDigraph {
  public:

    typedef SparseDigraph Digraph;
    typedef int Node;
    typedef int64_t Arc;

  private:

    int _node_num;
    int64_t _arc_num;

    std::vector<Node> _arc_sources; // _arc_sources[arc_id] = source node
    std::vector<Node> _arc_targets; // _arc_targets[arc_id] = target node

    std::vector<int64_t> _row_ptr;  // CSR row pointers (size: node_num+1)
    std::vector<Arc> _arc_ids;      // arc IDs in source order

    mutable std::vector<std::vector<Arc>> _in_arcs; // _in_arcs[node] = incoming arc IDs
    mutable bool _in_arcs_built;

    mutable std::vector<int64_t> _arc_to_out_pos; // _arc_to_out_pos[arc_id] = position in _arc_ids
    mutable std::vector<int64_t> _arc_to_in_pos;  // _arc_to_in_pos[arc_id] = position in _in_arcs[target]
    mutable bool _position_maps_built;

    void build_in_arcs() const {
      if (_in_arcs_built) return;

      _in_arcs.resize(_node_num);
      for (Arc a = 0; a < _arc_num; ++a) {
        _in_arcs[_arc_targets[a]].push_back(a);
      }
      _in_arcs_built = true;
    }

    void build_position_maps() const {
      if (_position_maps_built) return;

      _arc_to_out_pos.resize(_arc_num);
      _arc_to_in_pos.resize(_arc_num);

      for (int64_t pos = 0; pos < _arc_num; ++pos) {
        _arc_to_out_pos[_arc_ids[pos]] = pos;
      }

      build_in_arcs();
      for (int node = 0; node < _node_num; ++node) {
        const std::vector<Arc>& in = _in_arcs[node];
        for (size_t pos = 0; pos < in.size(); ++pos) {
          _arc_to_in_pos[in[pos]] = pos;
        }
      }

      _position_maps_built = true;
    }

  public:

    explicit SparseDigraph(int n)
      : _node_num(n), _arc_num(0),
        _in_arcs_built(false), _position_maps_built(false) {}

    void buildFromEdges(const std::vector<std::pair<Node, Node>>& edges) {
      _arc_num = edges.size();
      _arc_sources.resize(_arc_num);
      _arc_targets.resize(_arc_num);
      _arc_ids.resize(_arc_num);
      _in_arcs_built = false;
      _position_maps_built = false;
      _in_arcs.clear();
      _arc_to_out_pos.clear();
      _arc_to_in_pos.clear();

      // Create indexed edges: (source, target, original_arc_id)
      std::vector<std::tuple<Node, Node, Arc>> indexed_edges;
      indexed_edges.reserve(_arc_num);
      for (Arc i = 0; i < _arc_num; ++i) {
        indexed_edges.emplace_back(edges[i].first, edges[i].second, i);
      }

      // Sort by source node, then by target node (CSR requirement)
      std::sort(indexed_edges.begin(), indexed_edges.end(),
                [](const auto& a, const auto& b) {
                  if (std::get<0>(a) != std::get<0>(b))
                    return std::get<0>(a) < std::get<0>(b);
                  return std::get<1>(a) < std::get<1>(b);
                });

      _row_ptr.assign(_node_num + 1, 0);
      int current_row = 0;

      for (int64_t i = 0; i < _arc_num; ++i) {
        Node src = std::get<0>(indexed_edges[i]);
        Node tgt = std::get<1>(indexed_edges[i]);
        Arc orig_arc_id = std::get<2>(indexed_edges[i]);

        // Fill out row_ptr for rows with no outgoing edges
        while (current_row < src) {
          _row_ptr[++current_row] = i;
        }

        _arc_sources[orig_arc_id] = src;
        _arc_targets[orig_arc_id] = tgt;
        _arc_ids[i] = orig_arc_id;
      }

      // Fill remaining row_ptr entries
      while (current_row < _node_num) {
        _row_ptr[++current_row] = _arc_num;
      }
    }

    int nodeNum() const { return _node_num; }
    int64_t arcNum() const { return _arc_num; }

    int maxNodeId() const { return _node_num - 1; }
    int64_t maxArcId() const { return _arc_num - 1; }

    Node source(Arc arc) const { return _arc_sources[arc]; }
    Node target(Arc arc) const { return _arc_targets[arc]; }

    static int id(Node node) { return node; }
    static int64_t id(Arc arc) { return arc; }

    static Node nodeFromId(int id) { return Node(id); }
    static Arc arcFromId(int64_t id) { return Arc(id); }

    void first(Node& node) const { node = _node_num - 1; }
    static void next(Node& node) { --node; }

    void first(Arc& arc) const { arc = _arc_num - 1; }
    static void next(Arc& arc) { --arc; }

    void firstOut(Arc& arc, const Node& node) const {
      if (node < 0 || node >= _node_num) {
        arc = -1;
        return;
      }

      int64_t start = _row_ptr[node];
      int64_t end = _row_ptr[node + 1];

      arc = (start < end) ? _arc_ids[start] : Arc(-1);
    }

    void nextOut(Arc& arc) const {
      if (arc < 0) return;

      build_position_maps();

      int64_t pos = _arc_to_out_pos[arc];
      Node src = _arc_sources[arc];
      int64_t end = _row_ptr[src + 1];

      arc = (pos + 1 < end) ? _arc_ids[pos + 1] : Arc(-1);
    }

    void firstIn(Arc& arc, const Node& node) const {
      build_in_arcs();

      if (node < 0 || node >= _node_num) {
        arc = -1;
        return;
      }

      const std::vector<Arc>& in = _in_arcs[node];
      arc = in.empty() ? Arc(-1) : in[0];
    }

    void nextIn(Arc& arc) const {
      if (arc < 0) return;

      build_position_maps();

      int64_t pos = _arc_to_in_pos[arc];
      Node tgt = _arc_targets[arc];
      const std::vector<Arc>& in = _in_arcs[tgt];

      arc = (pos + 1 < static_cast<int64_t>(in.size())) ? in[pos + 1] : Arc(-1);
    }
  };

} //namespace lemon
