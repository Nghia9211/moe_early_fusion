# Dataset
- Lấy từ Amazon (AgentRecBench) : Musical Instruments và Industrial & Scientific

# Train Task


# Moe Task
- Thử fix để w/o seman không tốt hơn sau khi train 

# Plot Task 
- MoE Gating Weight của 3 expert (Box Plot)
- Lấy Category của items từng dataset để plot : ( Lấy các Category để làm cluster )
    + GCN Embeddings (TSNE) - Visualize user và Items Embedding theo category
    + Semantic Embeddings (TSNE) - Visualize Item theo category
    + SASrec Embeddings (TSNE) - Visualize Item theo category
- Biểu đồ phần trăm các case improve , drop , giữ nguyên khi dùng User Agent và Reranker (Pie Chart) 
- Rank Shift Distribution (của cả Reranker (Round 1) + User Agent (Round > 2)) : Trục X là Rank mới , Trục Y là Tần suất trên từng dataset + kịch bản (Chưa đủ log)
- Hiệu suất (NDCG/HR) thay đổi qua từng Vòng Feedback (Line Chart). (Chưa đủ log)
- Tương quan giữa Độ dài lịch sử (Theo bucket) với MoE distribution Weights (Scatter/Line) (Chưa đủ log)

- Đã có Dialouge Reranker và User Agent ( Có thể lấy ra để các case cụ thể)
