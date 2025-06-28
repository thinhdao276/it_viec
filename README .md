# 🏢 Company Recommendation System

A content-based recommendation system for companies using Natural Language Processing (NLP) techniques. This Streamlit application helps users find similar companies based on their descriptions, industries, and key skills.

## 🎯 Features

- **Company Similarity Search**: Find companies similar to a selected one
- **Text-based Search**: Search companies using custom text queries
- **Dual Recommendation Methods**: Compare results from Scikit-learn and Gensim approaches
- **Interactive Visualizations**: Charts and data exploration tools
- **User-friendly Interface**: Modern Streamlit web application

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Libraries**: Scikit-learn, Gensim
- **Data Processing**: Pandas, NumPy
- **Text Processing**: underthesea, googletrans
- **Visualization**: Plotly, Matplotlib, WordCloud

## 📊 How It Works

The system uses two main approaches for generating recommendations:

### 1. Scikit-learn TF-IDF + Cosine Similarity
- Uses sklearn's TfidfVectorizer to convert text to numerical features
- Calculates cosine similarity between company descriptions
- Fast and efficient for most use cases

### 2. Gensim TF-IDF + Cosine Similarity
- Uses Gensim's dictionary and corpus approach
- More memory efficient for large datasets
- Provides slightly different similarity calculations

## 🚀 Installation and Setup

1. **Clone the repository** (if applicable):
   ```bash
   cd Project2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**:
   - Ensure you have the `Overview_Companies.xlsx` file in the correct location
   - The app will look for it in `../Du lieu cung cap/Overview_Companies.xlsx`

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
Project2/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py            # Text preprocessing functions
│   ├── recommendation_sklearn.py   # Scikit-learn recommendation engine
│   ├── recommendation_gensim.py    # Gensim recommendation engine
│   └── visualization.py            # Visualization utilities
└── Content Based Suggestion.ipynb # Original Jupyter notebook
```

## 📊 Data Sources

The recommendation system analyzes three main data sources:

- **Company Overview**: General description of the company
- **Company Industry**: Business sector and domain
- **Key Skills**: Technical skills and technologies used

## 🔍 Usage Guide

### Company Similarity Search
1. Navigate to the "Company Similarity" tab
2. Select a company from the dropdown
3. Choose your preferred recommendation method
4. Adjust the number of recommendations
5. Click "Get Recommendations"

### Text-based Search
1. Go to the "Text-based Search" tab
2. Enter keywords, skills, or industry descriptions
3. Select your recommendation method
4. Click "Search Companies"

### Data Exploration
1. Visit the "Data Exploration" tab to:
   - View basic dataset statistics
   - Explore industry distribution
   - Browse sample data

## 📈 Use Cases

- **Job Seekers**: Find companies with similar tech stacks or industries
- **Business Development**: Identify potential partners or competitors
- **Market Research**: Analyze company landscapes and trends
- **Recruitment**: Discover companies with specific skill requirements

## ⚙️ Configuration

The application includes several configurable options:

- **Number of Recommendations**: 1-10 recommendations
- **Recommendation Method**: Choose between Scikit-learn, Gensim, or both
- **Search Query**: Custom text input for finding relevant companies

## 🐛 Troubleshooting

### Common Issues

1. **Data file not found**:
   - Ensure `Overview_Companies.xlsx` is in the correct location
   - Check file permissions

2. **Translation errors**:
   - The app uses Google Translate for Vietnamese text
   - Some translations may be rate-limited

3. **Memory issues**:
   - For large datasets, consider using Gensim over Scikit-learn
   - Reduce the number of companies loaded

### Performance Tips

- The application uses Streamlit's caching for better performance
- Models are built once and cached for subsequent use
- Data loading is optimized with caching

## 🤝 Contributing

This project was developed as part of a machine learning and NLP course. Contributions and improvements are welcome.

## 📝 License

This project is for educational purposes. Please ensure proper attribution when using or modifying the code.

## 🔗 Related Files

- **Original Notebook**: `Content Based Suggestion.ipynb`
- **Data Source**: `../Du lieu cung cap/Overview_Companies.xlsx`
- **Project Documentation**: Various PDF files in the parent directory

---

Built with ❤️ using Streamlit and modern NLP techniques.

# IT Viec dataset

ITViec là nền tảng chuyên cung cấp các cơ hội việc làm trong lĩnh vực Công nghệ Thông tin (IT) hàng đầu Việt Nam. Nền tảng này được thiết kế để giúp người dùng, đặc biệt là các developer, phát triển sự nghiệp một cách hiệu quả.
Người dùng có thể dễ dàng tìm kiếm việc làm trên ITViec theo nhiều tiêu chí khác nhau như kỹ năng, chức danh và công ty. Bên cạnh đó, ITViec còn cung cấp nhiều tài nguyên hữu ích hỗ trợ người tìm việc và phát triển bản thân, bao gồm:
• Đánh giá công ty: Giúp ứng viên có cái nhìn tổng quan về môi trường làm việc và văn hóa của các công ty IT.
• Blog: Chia sẻ các bài viết về kiến thức chuyên môn, kỹ năng mềm, xu hướng công nghệ và các lời khuyên nghề nghiệp hữu ích.
• Báo cáo lương IT: Cung cấp thông tin về mức lương trên thị trường, giúp người dùng có cơ sở để đàm phán mức đãi ngộ phù hợp.

2. MÔ TẢ BỘ DỮ LIỆU ITVIEC

Bộ dữ liệu ITViec được thu thập theo thứ tự tên của các công ty, bao gồm 3 files Excel:

Overview_Companies
File dữ liệu này chứa các thông tin tổng quan của các công ty, bao gồm:
• id: mã định danh của mỗi công ty.
• Company Name: tên công ty.
• Company Type: loại hình công ty (ví dụ: Outsourcing, Product, IT Service and IT Consulting,…)
• Company industry: lĩnh vực hoạt động của công ty (ví dụ: Software Development, Game, E-commerce,…)
• Company size: quy mô nhân sự của công ty (ví dụ: 1-50 nhân viên, 51-100 nhân viên,…)
• Country: quốc gia nơi công ty đặt trụ sở (ví dụ: Việt Nam, Singapore,…)
• Working days: các ngày làm việc trong tuần (ví dụ: Monday – Friday,…)
• Overtime Policy: chính sách làm thêm giờ của công ty (ví dụ: No OT, Extra salary for OT, …)
• Company Review: mô tả ngắn gọn về công ty, có thể bao gồm lĩnh vực hoạt động, sứ mệnh hoặc một vài thông tin nổi bật.
• Our key skills: liệt kê các kỹ năng (lập trình ngôn ngữ) chính mà công ty tìm kiếm ở ứng viên hoặc các kỹ năng nổi bật của đội ngũ nhân viên công ty
• Why you’ll love working here: nêu bật những lý do tại sao ứng viên nên làm việc tại công ty này, có thể liên quan đến văn hóa, chế độ đãi ngộ, cơ hội phát triển,…
• Location: địa điểm làm việc.
• Href: đường dẫn URL đến trang chi tiết của công ty trên nền tảng ITViec.

Overview_Reviews
File dữ liệu này chứa các đánh giá liên quan đến nhiều khía cạnh khác nhau của trải nghiệm làm việc tại các công ty, bao gồm:
• id: mã định danh của mỗi công ty.
• Company Name: tên công ty.
• Number of reviews: số lượng đánh giá mà công ty nhận được.
• Overall rating: điểm đánh giá tổng thể về công ty (thang điểm từ 0.0 đến 5.0).
• Salary & benefits: điểm đánh giá về mức lương và các phúc lợi của công ty (thang điểm từ 0.0 đến 5.0).
• Training & learning: điểm đánh giá về cơ hội đào tạo và phát triển tại công ty (thang điểm từ 0.0 đến 5.0).
• Management cares about me: điểm đánh giá về sự quan tâm của quản lý đối với nhân viên (thang điểm từ 0.0 đến 5.0)
Culture & fun: điểm đánh giá về văn hóa làm việc, văn hóa công ty và các hoạt động vui chơi, thể thao,… tại công ty (thang điểm từ 0.0 đến 5.0).
• Office & workspace: điểm đánh giá về môi trường công việc và không gian làm việc (thang điểm từ 0.0 đến 5.0).
• Recommend working here to a friend: tỷ lệ phần trăm nhân viên sẵn sàng giới thiệu bạn bè làm việc tại công ty này..

Lưu ý:
- Những công ty có điểm Overall rating bằng 0.0 thì sẽ không điểm Salary & benefits, Training & learning, Management cares about me, Culture & fun, Office & workspace và Recommend working here to a friend (do không có lượt đánh giá).
- Có những công ty có lượt đánh giá nhưng không có các chỉ số như Overall rating, Salary & benefits,… do các công ty này đã ẩn đi hết những lượt đánh giá.

Reviews
File dữ liệu này chứa các bình luận và đánh giá chi tiết của người dùng dành cho các công ty, bao gồm:
• id: mã định danh của mỗi công ty.
• Company Name: tên công ty được đánh giá.
• Cmt_day: thời gian mà người dùng bình luận.
• Title: tiêu đề ngắn gọn của bình luận
• What I liked: nội dung chi tiết của bình luận, tập trung vào những điều người dùng thích hoặc đánh giá cao về công ty.
• Suggestions for improvement: các ý kiến hoặc đề xuất để công ty có thể cải thiện.
• Rating: điểm đánh giá tổng quan của người dùng dành cho công ty (thang điểm từ 0 đến 5).
• Salary & benefits: điểm đánh giá về khía cạnh mức lương và các phúc lợi của người dùng dành cho công ty (thang điểm từ 0 đến 5).
• Training & learning: điểm đánh giá về khía cạnh cơ hội đào tạo và phát triển tại công ty của người dùng dành cho công ty (thang điểm từ 0 đến 5).
• Management cares about me: điểm đánh giá về khía cạnh quan tâm của quản lý đối với nhân viên của người dùng dành cho công ty (thang điểm từ 0 đến 5).
• Culture & fun: điểm đánh giá về khía cạnh văn hóa làm việc, văn hóa công ty và các hoạt động vui chơi, thể thao,… của người dùng dành cho công ty (thang điểm từ 0 đến 5).
• Office & workspace: điểm đánh giá về khía cạnh môi trường công việc và không gian làm việc của người dùng dành cho công ty (thang điểm từ 0 đến 5).
• Recommend?: Người đánh giá có giới thiệu công ty này cho người khác làm việc hay không (Yes hoặc No).


Yêu cầu 1: Dựa trên những thông tin từ các công ty đăng trên ITViec để gợi ý các công ty tương tự dựa trên nội dung mô tả. Content Based Simmilarity
Yêu cầu 2: Dựa trên những thông tin từ review của ứng viên/ nhân viên đăng trên ITViec để dự đoán khả năng “Recommend” công ty.
