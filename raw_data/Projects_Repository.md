PROJECTS REPOSITORY  
Suraj Kumar \- GitHub Portfolio

Comprehensive Technical Documentation of All Projects

═══════════════════════════════════════════════════════════

PROJECTS REPOSITORY  
Suraj Kumar \- GitHub Portfolio Documentation

═══════════════════════════════════════════════════════════════

PROJECT \#1

📌 Project Name: Sysmind-CLI

🌐 Domain / Field: Systems Programming, DevOps, System Administration, Command-Line Tools

🔗 GitHub Repository:  
https://github.com/Suraj-creation/Sysmind-CLI

🧠 Project Description:  
Sysmind-CLI is a unified command-line utility that provides intelligent system monitoring, process management, disk analytics, network diagnostics, and automated maintenance through a single cohesive interface. Unlike traditional fragmented system tools, Sysmind treats the system as an interconnected ecosystem, correlating resource usage, process behavior, and disk activity to provide actionable insights rather than just raw data. The tool addresses critical system administration challenges including system slowdowns, disk space management, network diagnostics, process management blindness, and risky cleanup operations. It employs a safety-first design with a quarantine system featuring full undo support for file operations.

Key features include: real-time system monitoring with historical baseline comparison, multi-phase duplicate file detection (size → quick hash → full hash), SQLite-based data persistence for trend analysis, cross-component intelligence correlation, health scoring algorithm (0-100), and platform abstraction layer supporting Windows, Linux, and macOS.

🛠️ Tech Stack:  
\- Programming Language: Python 3.8+  
\- Database: SQLite (built-in)  
\- Libraries: Python Standard Library only (no external dependencies)  
\- Architecture: Modular CLI with command hierarchy

🔧 Tools & Platforms Used:  
\- Python (https://www.python.org)  
\- SQLite (https://www.sqlite.org)  
\- Git version control  
\- Cross-platform OS APIs (WMI for Windows, /proc for Linux)

🚀 Deployment:  
Not deployed (CLI tool for local installation via pip install \-e .)

──────────────────────────────────────────────────────────

PROJECT \#2

📌 Project Name: Jarurat-Care

🌐 Domain / Field: Healthcare Technology, AI-powered Support Systems, Social Impact, Non-Profit Tech

🔗 GitHub Repository:  
https://github.com/Suraj-creation/Jarurat-Care

🧠 Project Description:  
Jarurat Care Foundation is a comprehensive digital platform serving as the technological backbone for an NGO dedicated to supporting cancer patients and their families throughout India. Founded in memory of Rekha Joshi (1963-2023), who lost her battle with Cholangiocarcinoma (bile duct cancer), the platform embodies the philosophy "Jaisi Jarurat, Vaisi Care" (As the Need, So the Care).

The platform provides: (1) Cancer patient support request forms with AI-powered suggestions covering multiple cancer types including gallbladder, breast, lung, cervical, oral, ovarian, prostate, and thyroid cancers; (2) Volunteer and mentor registration system for matching healthcare professionals, cancer survivors, and compassionate individuals with patients; (3) Hope \- an AI-powered chatbot built with Google Gemini AI (gemini-2.0-flash model) providing 24/7 cancer-specific support, empathetic responses, and resource recommendations; (4) Real-time analytics dashboard with AI-generated insights for administrators to understand community needs and optimize resource allocation.

The platform has facilitated support for 150+ cancer patients, coordinated 28+ doctors and 54+ mentors, and reached 2000+ community members. It integrates advanced AI prompt engineering tailored specifically for cancer support context, maintaining conversation history and providing context-aware responses.

🛠️ Tech Stack:  
\- Frontend: HTML5, CSS3, Vanilla JavaScript  
\- AI/ML: Google Gemini AI (gemini-2.0-flash model)  
\- Data Visualization: Chart.js 4.4.0  
\- Icons: Font Awesome 6.4.0  
\- Typography: Google Fonts (Poppins, Roboto)  
\- Data Storage: LocalStorage (demo), designed for MongoDB/PostgreSQL in production

🔧 Tools & Platforms Used:  
\- Google Gemini AI API (https://ai.google.dev/)  
\- Chart.js for analytics visualization (https://www.chartjs.org/)  
\- Vercel for deployment (https://vercel.com/)  
\- VS Code with Live Server for development

🚀 Deployment:  
https://jarurat-care-cyan.vercel.app/  
Live deployment on Vercel with production environment

──────────────────────────────────────────────────────────

PROJECT \#3

📌 Project Name: Machine\_learning (EEG-Based Alzheimer's Detection)

🌐 Domain / Field: Healthcare Technology, Machine Learning, Medical AI, Neuroscience, EEG Signal Processing

🔗 GitHub Repository:  
https://github.com/Suraj-creation/Machine\_learning

🧠 Project Description:  
An advanced machine learning platform for automated classification of Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal (CN) individuals using resting-state EEG biomarkers. The system implements a comprehensive pipeline encompassing EEG signal processing, feature extraction (438 biomarkers), and ensemble machine learning for clinical-grade diagnostic support.

The platform addresses the critical challenge of non-invasive, early detection of neurodegenerative diseases using electroencephalography (EEG) recordings. It processes EEG data from 88 subjects (36 AD, 23 FTD, 29 CN) acquired from OpenNeuro dataset ds004504 following the BIDS standard. The system achieves 72% accuracy in screening scenarios (Dementia vs Healthy), 77.8% AD recall, and 85.7% CN specificity.

Key technical features include: Multi-resolution Power Spectral Density (PSD) analysis using Welch's method (0.5-45 Hz), 438 engineered biomarkers spanning spectral/temporal/complexity domains, epoch augmentation (2-second windows with 50% overlap generating 4,400+ samples), ensemble architecture (LightGBM \+ XGBoost \+ Random Forest stacking), hierarchical classification strategy (binary specialists cascaded for differential diagnosis), and subject-level GroupKFold cross-validation preventing data leakage.

The interactive Streamlit web application features real-time inference (\<5s per subject), batch processing for multiple EEG files, comprehensive visualizations (PSD plots, topographic maps, confusion matrices, ROC curves), clinical interpretation dashboards with AI-generated insights, export capabilities (PDF reports, CSV features, JSON logs), and WCAG 2.1 accessibility compliance with dark mode support.

Clinical insights discovered: Peak Alpha Frequency (PAF) slowing in AD (8.06 Hz vs CN 8.30 Hz), elevated theta/alpha ratio as cognitive slowing marker (AD range 18-25 vs CN 3-17), occipital-temporal discrimination patterns, and enhanced frontal theta in FTD distinguishing it from AD.

🛠️ Tech Stack:  
\- Machine Learning: LightGBM 4.0+, XGBoost 2.0+, Random Forest, Scikit-learn  
\- Signal Processing: MNE-Python 1.5+ (EEG analysis framework), NumPy, SciPy  
\- Frontend: Streamlit 1.28+ (multi-page web application)  
\- Data Science: Pandas, Seaborn  
\- Visualization: Plotly (interactive charts), Matplotlib (PSD/topomaps), Chart.js 4.4.0  
\- Feature Engineering: Custom implementations (entropy, fractal dimension, connectivity metrics)  
\- Model Persistence: Joblib  
\- Reporting: ReportLab (PDF generation), Markdown  
\- Testing: Pytest  
\- Data Format: BIDS-compliant EEG data, EEGLAB .set format

🔧 Tools & Platforms Used:  
\- OpenNeuro (https://openneuro.org) \- Dataset ds004504 hosting  
\- MNE-Python (https://mne.tools) \- EEG processing  
\- Streamlit Cloud (https://streamlit.io/cloud) \- Deployment platform  
\- Vercel (https://vercel.com) \- Alternative deployment  
\- Docker \- Containerization  
\- Git LFS \- Large file storage for EEG datasets  
\- Jupyter Notebooks \- Research pipeline development  
\- VS Code with Dev Containers

🚀 Deployment:  
https://machine-learning-suraj-creation.streamlit.app/ (Primary)  
https://machine-learning-murex-two.vercel.app/ (Alternative)  
Full production deployment with 14 deployment instances, health monitoring, and auto-scaling

──────────────────────────────────────────────────────────

[https://github.com/Suraj-creation/DL\_course-Shabbeer.Basha](https://github.com/Suraj-creation/DL_course-Shabbeer.Basha)  
🎓 Project Description:  
A comprehensive full-stack educational platform designed for course instructors to create, manage, and publish course content through an intuitive admin dashboard. The platform provides a complete Learning Management System (LMS) with separate admin and public interfaces, enabling seamless course delivery and student engagement.

The system implements a dual-interface architecture: an administrative backend for content management and a public-facing website for student access. The platform supports comprehensive course management including lectures, assignments, teaching assistants, tutorials, prerequisites, exams, and resource organization.

Key features include: Admin authentication with JWT-based security, course management (create/edit course information, metadata, settings), lecture management with multi-format support (YouTube video links, PDF/PPT slides, reading materials), assignment system with due dates and grading templates, teaching assistant directory with contact details and office hours, tutorial management with supplementary materials, prerequisite tracking with external resource links, exam scheduling, and categorized resource library (Books, Tools, Papers).

The architecture consists of 8 MongoDB database models (Admin, Course, Lecture, Assignment, TeachingAssistant, Tutorial, Prerequisite, Exam, Resource) with 9 RESTful API route groups, file upload middleware with Multer supporting PDFs, PPTs, and documents, real-time content updates with instant publish/unpublish functionality, and comprehensive CRUD operations for all content types.

The public website features clean navigation structure, responsive mobile-friendly design, course overview with hero sections, organized curriculum view with all lecture materials, assignment tracking with status indicators, TA directory with office hours, and categorized resource library.🛠️ Tech Stack:  
\- Backend: Node.js, Express.js (RESTful API), MongoDB (NoSQL database), Mongoose (ODM), JWT (JSON Web Tokens for authentication), bcryptjs (password hashing), Multer (file upload middleware)  
\- Frontend: React.js 18+ (UI framework with hooks), React Router 6+ (client-side routing), Axios (HTTP client for API calls), React Icons (icon library), CSS3 (custom styling with responsive design)  
\- Security: JWT token-based authentication, bcrypt password encryption, Protected API routes with auth middleware, File type validation, File size limits (10MB default), XSS protection, CORS configuration  
\- Database Models: 8 comprehensive schemas (Admin, Course, Lecture, Assignment, TeachingAssistant, Tutorial, Prerequisite, Exam, Resource)  
\- API Architecture: 9 RESTful route groups with full CRUD operations, Auth routes (login, register, password change), Course/Lecture/Assignment/TA/Tutorial/Prerequisite/Exam/Resource routes  
\- File Handling: Multer middleware configuration, Support for PDFs, PPTs, DOCs, images, Upload directory management, File size and type validation

🔧 Tools & Platforms Used:  
\- MongoDB Atlas \- Cloud database hosting  
\- Node.js Runtime \- JavaScript execution environment  
\- Express.js Framework \- Web application framework  
\- React Development Tools \- Component development and debugging  
\- Postman \- API testing and development  
\- Git & GitHub \- Version control and collaboration  
\- VS Code \- Primary development environment  
\- npm/yarn \- Package management

🚀 Deployment:  
Production deployment with 58 deployment instances on Vercel  
Frontend: Vercel (React build with static file serving)  
Backend: Vercel serverless functions / Railway  
Database: MongoDB Atlas cloud database  
Responsive design optimized for mobile (\<768px), tablet (768-1024px), and desktop (\>1024px) breakpoints  
Project Status: \~75% complete with production-ready backend, complete admin panel structure, and templates for remaining features

─────────────────────────────────────────────────────────

[https://github.com/Suraj-creation/Sysmind-CLI](https://github.com/Suraj-creation/Sysmind-CLI)  
🧠 Project Description:  
SYSMIND (System Intelligence & Automation CLI) is a unified command-line utility that provides intelligent system monitoring, process management, disk analytics, network diagnostics, and automated maintenance through a single cohesive interface. Unlike traditional system tools that operate in isolation, SYSMIND treats the system as an interconnected ecosystem, correlating resource usage, process behavior, and disk activity to provide actionable insights rather than just raw data.

The platform addresses critical challenges that developers, students, and power users face daily: system slowdown without clear cause, mysterious disk space disappearance, network issues without diagnostic clarity, process management blindness, and risky cleanup operations. SYSMIND solves these through unified system view, historical intelligence learning from past data, safety-first design with quarantine system and full undo support, actionable insights beyond raw data, and cross-correlation linking events across different system components.

Key features include: System Monitor with real-time CPU/memory/disk I/O monitoring, historical data tracking with trend analysis, baseline establishment and anomaly detection, and live dashboard; Disk Intelligence with space analysis and visual breakdowns, multi-phase duplicate detection (size → quick hash → full hash), safe cleanup with quarantine system, and large/old file finder; Process Manager with detailed process listing and context, process tree visualization, startup program management, and watchdog rules for automated monitoring; Network Diagnostics with connectivity testing (DNS, gateway, internet), active connection monitoring, bandwidth usage by process, and listening ports analysis; Intelligence Core with overall system health scoring (0-100), cross-component correlation, anomaly detection, and AI-driven recommendations.

Technical innovation: Uses only Python standard library (no external dependencies) for maximum portability and zero configuration, SQLite database for all data storage with ACID compliance, multi-phase duplicate detection algorithm (O(n) size scan → O(k) quick hash → O(m) full SHA-256), quarantine system that never permanently deletes files directly (30-day retention before permanent deletion), weighted health score algorithm across 5 components (CPU 25%, Memory 25%, Disk 20%, Process 15%, Network 15%), and platform abstraction layer for Windows/Linux/macOS support.

🛠️ Tech Stack:  
\- Core: Python 3.8+ (standard library only \- zero external dependencies)  
\- Database: SQLite3 (built-in, ACID compliant, zero configuration)  
\- System Monitoring: psutil-alternative using /proc, ctypes, WMI for cross-platform metrics  
\- Data Storage: JSON configuration files, SQLite database with indexed queries  
\- CLI Framework: argparse (standard library) for command parsing and routing

\- Hashing: hashlib SHA-256 for duplicate detection- Platform Support: Windows (WMI/Registry), Linux (/proc filesystem), macOS (system calls)

🔧 Tools & Platforms Used:  
\- Python 3.8+ Runtime \- Cross-platform execution environment  
\- SQLite3 \- Embedded database engine (part of Python standard library)  
\- Git & GitHub \- Version control and collaboration  
\- VS Code \- Primary development environment  
\- Virtual Environments \- Isolated development with venv  
\- pip \- Package distribution (development mode installation)

🚀 Deployment:  
Local installation via pip (pip install \-e .)  
Command-line tool available globally after installation  
Data stored in \~/.sysmind/ directory (database, config, logs, quarantine)  
Cross-platform support: Windows, Linux, macOS  
Zero external dependencies \- works offline without pip after installation  
Project Status: Version 1.0.0, fully functional with comprehensive documentation, MIT License

─────────────────────────────────────────────────────────

[https://github.com/Suraj-creation/Jarurat-Care](https://github.com/Suraj-creation/Jarurat-Care)

💜 Project Description:

Jarurat Care Foundation is a Cancer Support Platform founded in loving memory of Rekha Joshi (1963-2023), who lost her battle with Cholangiocarcinoma (bile duct cancer). Created by her daughter Priyanka Joshi along with Ayush Anand, the platform transforms personal grief into purpose by ensuring no cancer patient or family faces this battle alone. The name embodies the philosophy 'Jaisi Jarurat, Vaisi Care' (As the Need, So the Care), recognizing that every patient's needs are unique and tailoring support accordingly.The NGO-backed digital platform provides comprehensive support services including Patient Advocacy (navigating healthcare systems and ensuring patient rights), Access to Treatment (connecting patients with oncologists, hospitals, and clinical trials), Holistic Care (emotional, psychological, and nutritional support), Educational Resources (cancer awareness, prevention, and survivor stories), and Community Connection (peer mentorship from cancer survivors). The platform has assisted 150+ patient families, built a network of 28+ doctors and oncologists, engaged 54+ mentors (cancer survivors and counselors), and reached 2000+ community members.

Key platform features include: Hope AI Chatbot powered by Google Gemini (gemini-2.0-flash) providing 24/7 cancer-specific support with empathetic responses, cancer FAQ database, and quick links to resources; Comprehensive Patient Support Request Form for cancer patients to request help with cancer type selection (covering gallbladder, breast, lung, cervical, oral, ovarian, prostate, thyroid and more), support type needed (emotional, financial, treatment navigation, peer mentorship, nutrition), and AI-powered field suggestions; Volunteer & Mentor Registration system for cancer survivors, healthcare professionals, and compassionate individuals to register with skill matching and availability; Analytics Dashboard with real-time insights, patient statistics by cancer type, volunteer metrics, AI-generated recommendations and trends; Contact & Communication system with direct forms, emergency helpline, and automated follow-ups.

AI Integration Architecture: The platform leverages Google Gemini AI for intelligent automation with Hope Chatbot using cancer-specific system prompts, Dashboard AI Insights analyzing data to generate actionable recommendations (e.g., 'High demand for emotional support in gallbladder cancer patients', 'Need more Gujarati-speaking volunteers in Mumbai'), Form AI Suggestions for smart field completion, and centralized AI Service Module with rate limiting, error handling, context management, and prompt engineering optimized for cancer support context.

🛠️ Tech Stack:  
\- Frontend: HTML5, CSS3, Vanilla JavaScript (ES6+)  
\- AI Layer: Google Gemini AI API (gemini-2.0-flash model), GeminiAIService class for API communication, Cancer-specific prompt engineering, Context-aware response generation  
\- Data Layer: LocalStorage (demo implementation), Sample Data Generator for testing  
\- Visualization: Chart.js 4.4.0 for data visualization and analytics charts  
\- Icons & Typography: Font Awesome 6.4.0, Google Fonts (Poppins, Roboto)  
\- Design Principles: Accessibility-first (WCAG compliant, screen reader friendly), Mobile responsive, Progressive enhancement, Empathetic design with calming colors and supportive messaging

🔧 Tools & Platforms Used:  
\- Google Gemini AI \- Advanced AI model for cancer support chatbot  
\- Chart.js \- Data visualization library  
\- Font Awesome \- Icon library  
\- Google Fonts \- Typography system  
\- Git & GitHub \- Version control and collaboration  
\- Vercel \- Hosting and deployment platform

🚀 Deployment:  
[https://jarurat-care-cyan.vercel.app/](https://jarurat-care-cyan.vercel.app/)  
Deployed on Vercel with continuous deployment  
Serves static frontend with AI API integration  
Project Status: Active NGO platform serving cancer patients and families, founded in 2024 in memory of Rekha Joshi  
Founders: Priyanka Joshi (Co-Founder & Director), Ayush Anand (Co-Founder), Advisory: Dr. Chetan Arora (IIT Delhi)

─────────────────────────────────────────────────────────

[https://github.com/Suraj-creation/chatgpt\_clone](https://github.com/Suraj-creation/chatgpt_clone)  
🤖 Project Description:  
Gemini Chat UI is a production-quality web application that replicates the ChatGPT interface using Google's Gemini API. Built with modern web technologies (Next.js 14, TypeScript, Tailwind CSS), it provides a familiar and intuitive chat interface for interacting with Google's Gemini AI models. The application offers a complete ChatGPT-like experience with conversation management, real-time streaming responses, and local data persistence.

Core functionality includes: Real-time streaming with token-by-token response streaming from Gemini API for smooth, natural conversation flow; Comprehensive conversation management allowing users to create, save, load, and delete conversations with automatic localStorage persistence; Model selection capability to switch between Gemini 1.5 Flash (fast and efficient), Gemini 1.5 Pro (most capable), and legacy Gemini 1.0 Pro; Beautiful dark theme interface inspired by ChatGPT's design language; Full markdown rendering with syntax highlighting for code blocks and technical content.

Advanced features: System instructions customization allowing users to modify AI behavior with custom prompts; One-click code copying from code blocks for easy code reuse; Keyboard shortcuts including Enter to send messages and Shift+Enter for new lines; Fully responsive design that works seamlessly across mobile, tablet, and desktop devices; Graceful error handling and recovery with user-friendly messages; localStorage-based persistence ensuring all conversations are saved locally without requiring server-side database.

🛠️ Tech Stack:

\- Framework: Next.js 14.2.0 (App Router, Server-Sent Events), React 18.3.0- Language: TypeScript 5.3.0 for type safety and better development experience  
\- Styling: Tailwind CSS 3.4.0 with custom dark theme configuration  
\- AI Integration: Google Gemini API (gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro models)  
\- Markdown: react-markdown 9.0.1 for rendering formatted text  
\- Code Highlighting: react-syntax-highlighter 15.5.0 for beautiful code displays

\- Icons: lucide-react 0.344.0 for consistent icon design- State Management: React Context API for global chat state  
\- Storage: Browser localStorage for conversation persistence

🔧 Tools & Platforms Used:  
\- Google Gemini API \- AI model integration  
\- Vercel \- Hosting and deployment platform  
\- Git & GitHub \- Version control (12 commits)  
\- VS Code \- Primary development environment  
\- Node.js 18+ \- Runtime environment

🚀 Deployment:  
[https://chatgpt-clone-taupe-one.vercel.app/](https://chatgpt-clone-taupe-one.vercel.app/)  
Deployed on Vercel with 20 deployment instances  
Continuous deployment from GitHub repository  
Environment variables configured for API key management  
Project Status: Production-ready application with comprehensive documentation, MIT License

─────────────────────────────────────────────────────────

[https://github.com/Suraj-creation/Snake-and-Ladder-game](https://github.com/Suraj-creation/Snake-and-Ladder-game)  
🧠 Project Description:  
Snake-and-Ladder-game is a personal project dedicated to Reesu and Reetu. Built using Google AI Studio's Gemini integration, this application showcases the integration of AI capabilities with interactive game development. The project is designed as a web-based gaming application that can be run locally or deployed to the cloud. It demonstrates the use of modern TypeScript development practices with AI-powered features through the Gemini API. The project structure follows Google AI Studio's app template, providing a solid foundation for building AI-enhanced interactive applications with proper component organization and configuration management.

🛠️ Tech Stack:  
\- TypeScript \- Type-safe programming language for application logic  
\- React (via Vite) \- Frontend framework for building interactive UI  
\- Google AI Studio \- Platform for AI-powered application development  
\- Google Gemini API \- AI model integration for intelligent features  
\- Vite \- Modern frontend build tool and dev server  
\- Node.js \- Runtime environment for development

🔧 Tools & Platforms Used:  
\- Google Gemini API \- AI model integration  
\- Google AI Studio \- Development and deployment platform  
\- npm \- Package management  
\- Git & GitHub \- Version control (3 commits)

🚀 Deployment:  
[https://ai.studio/apps/drive/17Q7-FDdkLie7JlnYBk0NJx0Bic9uAKeN](https://ai.studio/apps/drive/17Q7-FDdkLie7JlnYBk0NJx0Bic9uAKeN)  
Hosted on Google AI Studio platform  
Local development support with npm run dev  
Environment configuration via .env.local for Gemini API key  
Project Status: Active development \- Personal project for family members, AI-powered game application

─────────────────────────────────────────────────────────────────

[https://github.com/Suraj-creation/Portfolio\_finance\_Optimal](https://github.com/Suraj-creation/Portfolio_finance_Optimal)  
🧠 Project Description:  
Portfolio\_finance\_Optimal is a sophisticated web-based portfolio optimization dashboard implementing Modern Portfolio Theory (MPT) with real data from the National Stock Exchange of India. The application features an Excel-inspired interface with comprehensive visualizations of portfolio analytics and AI-powered insights using Google Gemini 1.5 Pro API. It analyzes 8 NSE assets from 2020-2024, providing optimal portfolio allocation that improves Sharpe ratio by 15.6% over equal-weighted distribution. The project includes detailed Jupyter notebook research (2,243 lines), comprehensive markdown documentation (2,978 lines), and a complete Excel workbook with 11 sheets containing historical data, optimization results, and efficient frontier calculations. Key features include SLSQP optimization for maximum Sharpe ratio, interactive visualizations using Plotly.js and ECharts, correlation matrices, Monte Carlo simulations, and export capabilities for Excel and PDF formats.

🛠️ Tech Stack:  
\- Python 3.14 \- Data analysis and optimization research  
\- pandas 2.3.3 \- Data manipulation and analysis  
\- numpy 2.3.4 \- Numerical computations and matrix operations  
\- yfinance \- Yahoo Finance API integration for NSE data  
\- scipy.optimize \- SLSQP optimization algorithm implementation  
\- HTML5 & JavaScript ES6+ \- Frontend application logic  
\- TailwindCSS \- Utility-first styling with Excel theme colors  
\- Plotly.js v3.0.3 \- Interactive charts (heatmaps, scatter plots)  
\- ECharts v5.4.3 \- Additional data visualizations  
\- Google Gemini 1.5 Pro API \- AI-powered portfolio insights  
\- Jupyter Notebook \- Research and documentation environment

🔧 Tools & Platforms Used:  
\- Yahoo Finance API \- Historical NSE price data (2020-2024)  
\- Google Gemini API \- AI insights generation  
\- Git & GitHub \- Version control (1 commit)  
\- Font Awesome v6.4.0 \- Icon library  
\- Excel \- Data workbook with 11 sheets (MPT\_Portfolio\_Results.xlsx)  
\- Modern Portfolio Theory (MPT) \- Markowitz optimization framework

🚀 Deployment:  
[https://portfolio-finance-optimal.vercel.app](https://portfolio-finance-optimal.vercel.app)  
Deployed on Vercel with 1 deployment instance  
Local development support with browser-based HTML interface  
Client-side JavaScript execution, no backend server required  
Portfolio Data: 8 NSE assets with 1,237 historical observations  
Optimization Results: 15.6% Sharpe ratio improvement (1.192 to 1.377)  
Project Status: Production-ready with comprehensive documentation (2,978 lines MD \+ 2,243 lines Jupyter notebook)

─────────────────────────────────────────────────────────────────

[https://github.com/Suraj-creation/Live\_Classroom-powered\_by\_AI](https://github.com/Suraj-creation/Live_Classroom-powered_by_AI)  
🧠 Project Description:  
Live\_Classroom-powered\_by\_AI, also known as ExplainBoard, is an AI-powered visual learning whiteboard that combines interactive education with cutting-edge AI technology. The application features two primary modes: Explain a Topic mode for generating comprehensive educational content with AI-generated illustrations, and Live Classroom mode for real-time speech-to-text transcription with dynamic visual explanations. Built with React 19 and TypeScript, the platform leverages Google Gemini 2.5 Pro API for content generation and Gemini 2.5 Flash for image creation. The application features a professional dark theme with classroom chalkboard aesthetics, real-time transcription processing every 7 seconds, and comprehensive export capabilities (PNG, PDF, Markdown). With production-ready performance metrics (106.25 KB gzipped build size), full mobile responsiveness, and 4 successful Vercel deployments, the project demonstrates advanced integration of AI models for educational purposes with seamless user experience.

🛠️ Tech Stack:  
\- React 19 \- Modern frontend framework for component-based UI  
\- TypeScript \- Type-safe programming for robust application logic  
\- Tailwind CSS \- Utility-first styling with dark theme and classroom aesthetics  
\- Vite \- Fast build tool and development server  
\- Google Gemini 2.5 Pro API \- Content generation and educational explanations  
\- Google Gemini 2.5 Flash \- AI image generation for illustrations  
\- Gemini Native Audio API \- Real-time speech-to-text transcription (PCM 16-bit 16kHz)  
\- Poppins & JetBrains Mono \- Modern typography for professional UI  
\- Web Audio API \- Real-time audio processing for live classroom

🔧 Tools & Platforms Used:  
\- Google Gemini API \- Multi-modal AI for text and image generation  
\- npm \- Package management and dependency handling  
\- Git & GitHub \- Version control (7 commits)  
\- Vercel \- Deployment platform with continuous integration  
\- PostCSS & Autoprefixer \- CSS processing and optimization

🚀 Deployment:  
[https://live-classroom-powered-by-ai.vercel.app](https://live-classroom-powered-by-ai.vercel.app)  
Deployed on Vercel with 4 successful deployment instances  
Local development: npm install && npm run dev  
Environment configuration: VITE\_GEMINI\_API\_KEY required in .env.local  
Performance Metrics: 106.25 KB gzipped, \<1s first paint, \<2s interactive  
Browser Compatibility: Full support for Chrome, Firefox, Safari, Edge, and mobile browsers  
Project Status: Production-ready with comprehensive UI/UX guide, deployment documentation, and testing checklist

─────────────────────────────────────────────────────────────────

[https://github.com/Suraj-creation/Image\_captioning\_-\_Segmentation](https://github.com/Suraj-creation/Image_captioning_-_Segmentation)  
🧠 Project Description:  
Image\_captioning\_-\_Segmentation is a production-quality Streamlit web application that combines image captioning and segmentation using state-of-the-art deep learning models trained on the COCO 2014 dataset. The application features dual functionality with multiple model options: ResNet50+LSTM and InceptionV3+Transformer for captioning, and Mask R-CNN, DeepLabV3+, and U-Net for segmentation. Built with comprehensive testing, CI/CD pipelines, and Docker support for both CPU and GPU environments, the project demonstrates professional software engineering practices. Key features include batch processing capabilities, interactive Streamlit UI with real-time visualization, combined pipeline for synchronized caption and segmentation with object linking, beam search with adjustable parameters (width 1-5, max length 10-30), and comprehensive export utilities. The application supports multiple upload methods (file upload, COCO 2014 samples, URL), includes developer mode for debugging, and provides detailed metrics calculation (BLEU, CIDEr). With 9 Vercel deployments (3 production), MIT license, and 89.6% Python codebase, the project showcases advanced computer vision and NLP integration.

🛠️ Tech Stack:  
\- Python 3.10+ \- Primary programming language  
\- PyTorch \- Deep learning framework for model implementation  
\- Streamlit \- Interactive web application framework  
\- ResNet50 & InceptionV3 \- CNN encoders for feature extraction  
\- LSTM & Transformer \- Sequence models for caption generation  
\- Mask R-CNN \- Instance segmentation model (COCO pretrained)  
\- DeepLabV3+ \- Semantic segmentation with atrous convolution  
\- U-Net \- Encoder-decoder architecture for segmentation  
\- NLTK \- Natural Language Toolkit for text processing  
\- OpenCV \- Computer vision preprocessing and visualization  
\- Docker & Docker Compose \- Containerization with CPU/GPU support

🔧 Tools & Platforms Used:  
\- COCO 2014 Dataset \- Training data for captioning and segmentation  
\- PyTorch Hub \- Pretrained model weights download  
\- Hugging Face \- Transformer model integration  
\- pytest \- Testing framework with coverage reporting  
\- GitHub Actions \- CI/CD pipeline automation  
\- Git & GitHub \- Version control (5 commits)  
\- Vercel \- Deployment platform

🚀 Deployment:  
[https://image-captioning-segmentation-nu.vercel.app](https://image-captioning-segmentation-nu.vercel.app)  
9 Vercel deployments (3 Production instances: image-captioning-segmentation-ab9r, image-captioning-segmentation, Production)  
Docker deployment: docker-compose up app (CPU) or docker-compose \--profile gpu up app-gpu (GPU)  
Local development: streamlit run app.py (available at [http://localhost:8501](http://localhost:8501))  
Project Structure: Modular design with models/, inference/, utils/, static/, tests/ directories  
Project Status: Production-ready with MIT License, comprehensive documentation (README, PROJECT\_SUMMARY, DEPLOYMENT, CHANGELOG, CONTRIBUTING), CI/CD pipeline, and testing suite

─────────────────────────────────────────────────────────────────

**🏥 Healthcare AI Assistant \- Intelligent Disease Prediction System**  
**🔗 Repository: [https://github.com/Suraj-creation/Healthcare\_Prediction](https://github.com/Suraj-creation/Healthcare_Prediction)**

**📝 Description:**  
**A cutting-edge, AI-powered healthcare diagnostic assistant that combines machine learning with Google Gemini AI to provide intelligent disease predictions, personalized health recommendations, and real-time medical insights based on symptoms. This educational tool helps users understand potential health conditions through advanced symptom analysis and natural language AI interactions.**

**✨ Key Features:**  
**\- Smart Symptom Checker \- Multi-modal input (text, voice, interactive body map)**  
**\- 41 Disease Models \- Machine learning-based disease prediction with confidence scoring**  
**\- 132 Symptoms Database \- Comprehensive symptom library with severity weighting (1-7 scale)**  
**\- Google Gemini 2.5 Flash AI \- Natural language Q\&A and personalized health insights**  
**\- Interactive Visualizations \- ECharts-powered charts for symptom analysis and recovery timelines**  
**\- Personalized Health Recommendations \- Custom medication plans, diet suggestions, and exercise routines**  
**\- Real-Time Validation \- Instant feedback and intelligent symptom suggestions**  
**\- Voice Input Support \- Hands-free symptom reporting via Web Speech API**  
**\- Privacy-First Design \- All data stored locally, no backend server required**  
**\- Responsive Modern UI \- Glassmorphism design with smooth Anime.js animations**

**💻 Technologies Used:**  
**\- HTML5 \- Semantic markup with accessibility features**  
**\- CSS3 & Tailwind CSS \- Modern responsive layouts with utility-first styling**  
**\- JavaScript ES6+ \- Modern JavaScript with async/await, modules, and classes**  
**\- Google Gemini 2.5 Flash \- AI model for natural language health insights**  
**\- Anime.js v3.x \- Smooth animations and micro-interactions**  
**\- ECharts.js v5.x \- Interactive charts and data visualization**  
**\- Typed.js v2.x \- Typewriter effects for AI responses**  
**\- Splitting.js v1.x \- Advanced text animations**  
**\- Web Speech API \- Voice recognition for symptom input**  
**\- Local Storage API \- Client-side data persistence**

**🛠️ Tools & Platforms Used:**  
**\- Google Gemini API \- AI insights generation and natural language processing**  
**\- GitHub \- Version control and repository hosting (3 months ago, initial commit)**  
**\- Vercel \- Potential deployment platform**  
**\- VS Code \- Development environment**  
**\- Python HTTP Server \- Local development testing**  
**\- Browser DevTools \- Testing and debugging**  
**\- COCO 2014 Dataset influence \- Medical symptom classification patterns**  
**\- WCAG 2.1 AA \- Accessibility compliance standards**

**🚀 Deployment:**  
**Local development ready with Python HTTP server or any static file server**  
**Client-side only application \- no backend server required**  
**Can be deployed to Vercel, Netlify, or GitHub Pages**  
**Requires Google Gemini API key configuration (free tier available)**  
**File structure: 5 HTML pages (index, results, recommendations, profile, interaction)**  
**Core logic: main.js (645 lines) with modular architecture**  
**Browser compatibility: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+**  
**Project Structure: Comprehensive documentation (README, AI\_TROUBLESHOOTING\_GUIDE, QUICK\_FIX\_GUIDE, GEMINI\_AI\_INTEGRATION, etc.)**  
**Project Status: Production-ready educational tool with MIT License, privacy-first design (Version 2.0, Last Updated: October 2025\)**

---

🧠 SYSMIND \- System Intelligence & Automation CLI  
🔗 Repository: [https://github.com/Suraj-creation/Sysmind-CLI](https://github.com/Suraj-creation/Sysmind-CLI)

📝 Description:  
A unified command-line utility that provides intelligent system monitoring, process management, disk analytics, network diagnostics, and automated maintenance through a single cohesive interface. Unlike traditional fragmented system tools, SYSMIND treats the system as an interconnected ecosystem with historical context, automatic cross-component analysis, and proactive baselines for anomaly detection.

✨ Key Features:  
\- System Monitor \- Real-time CPU, memory, disk I/O monitoring with historical data and trend analysis  
\- Disk Intelligence \- Space analysis with visual breakdowns and multi-phase duplicate detection  
\- Process Manager \- Process listing with detailed context, tree visualization, and startup program management  
\- Network Diagnostics \- Connectivity testing (DNS, gateway, internet) and active connection monitoring  
\- Intelligence Core \- System health scoring (0-100) with cross-component correlation and anomaly detection  
\- Safe Cleanup \- Quarantine system with full undo support instead of risky permanent deletions  
\- Baseline Establishment \- Learning from past data to provide context and detect abnormal behavior  
\- Automated Maintenance \- Watchdog rules for automated system monitoring  
\- Configuration Management \- JSON-based persistent settings with customizable thresholds  
\- Visual Dashboard \- Live monitoring interface with progress bars and actionable insights

💻 Technologies Used:  
\- Python 3.8+ \- Standard library only (no external dependencies for maximum portability)  
\- SQLite \- ACID compliant database for system snapshots, baselines, alerts, and quarantine manifests  
\- Platform Abstraction Layer \- OS-specific adapters for Windows (WMI), Linux (/proc), and macOS  
\- SHA-256 Hashing \- Multi-phase duplicate detection algorithm (size → quick hash → full hash)  
\- JSON Configuration \- Persistent settings management with customizable thresholds

🛠️ Tools & Platforms Used:  
\- GitHub \- Version control and repository hosting (2 commits, updated 12 hours ago)  
\- pip \- Python package installation in development mode (pip install \-e .)  
\- Virtual Environment \- Isolated Python environment for development  
\- Cross-platform Support \- Windows, Linux, and macOS compatibility

🚀 Deployment:  
Local development with virtual environment setup  
Installable via pip in development mode (pip install \-e .)  
Command-line interface available globally as 'sysmind' command  
Data storage in \~/.sysmind/ directory (SQLite database, config.json, logs, quarantine)  
No external dependencies or backend server required  
Platform Support: Windows, Linux, macOS  
Python 3.8+ requirement for standard library features  
Project Structure: Modular design with core/, modules/, commands/, and utils/ directories  
Project Status: Active development with MIT License, comprehensive documentation (README.md, manual.md, plan.md)

---

**♥️ Jarurat Care Foundation \- Cancer Support Platform**  
**🔗 Repository: https://github.com/Suraj-creation/Jarurat-Care**  
**🌐 Live Website: https://jarurat-care-cyan.vercel.app**

**📝 Description:**  
**A meaningful digital platform serving as the technological backbone for Jarurat Care Foundation, an NGO dedicated to supporting cancer patients and their families. Founded in memory of Rekha Joshi (1963-2023) who battled Cholangiocarcinoma, the platform embodies "Jaisi Jarurat, Vaisi Care" (As the Need, So the Care). The platform streamlines patient support requests, volunteer coordination, and provides 24/7 AI-powered assistance through Hope chatbot.**

**✨ Key Features:**  
**\- Hope AI Chatbot \- 24/7 cancer support assistant powered by Google Gemini 2.0 Flash with cancer-specific knowledge**  
**\- Patient Support Request Form \- Comprehensive intake for cancer patients covering 10+ cancer types**  
**\- Volunteer & Mentor Registration \- Skill matching system for oncologists, survivors, counselors, and social workers**  
**\- Analytics Dashboard \- Real-time insights with AI-generated recommendations and patient statistics**  
**\- Multi-Support Categories \- Emotional support, financial aid, treatment navigation, peer mentorship, nutrition counseling**  
**\- Cancer Type Coverage \- Specialized support for 10+ cancer types including Cholangiocarcinoma, breast, lung, cervical**  
**\- Impact Metrics \- 150+ patients assisted, 54+ mentors, 28+ doctors, 2000+ community reach**  
**\- AI-Powered Insights \- Predictive analytics for resource allocation and understaffed area identification**  
**\- Empathetic Design \- WCAG compliant, calming colors, supportive messaging for patients**  
**\- Contact Integration \- Direct communication with foundation team and emergency helpline**

**💻 Technologies Used:**  
**\- JavaScript ES6+ \- Vanilla JavaScript for frontend logic and AI integration**  
**\- HTML5 & CSS3 \- Semantic markup with modern styling and accessibility features**  
**\- Google Gemini 2.0 Flash \- AI model for Hope chatbot and dashboard insights**  
**\- Chart.js 4.4.0 \- Interactive data visualization for analytics dashboard**  
**\- Font Awesome 6.4.0 \- Icon library for UI elements**  
**\- Google Fonts \- Poppins and Roboto typography**  
**\- LocalStorage API \- Client-side data persistence for demo purposes**

**🛠️ Tools & Platforms Used:**  
**\- GitHub \- Version control and repository hosting (3 commits, updated yesterday)**  
**\- Vercel \- Deployment platform for production hosting**  
**\- Google Gemini API \- AI integration for chatbot and analytics insights**  
**\- Chart.js \- Data visualization library**  
**\- VS Code \- Development environment**  
**\- Modern Web Browsers \- Chrome, Firefox, Edge, Safari compatibility**

**🚀 Deployment:**  
**https://jarurat-care-cyan.vercel.app**  
**1 Vercel deployment with production hosting**  
**Local development with Python HTTP server or VS Code Live Server**  
**Client-side application with LocalStorage for demo data**  
**Google Gemini API integration requires API key configuration**  
**Mobile responsive design with accessibility-first approach**  
**File structure: 5 HTML pages (index, patient-form, volunteer-form, dashboard, contact)**  
**Project Structure: css/, js/ (ai-service.js, chatbot.js, dashboard.js, forms.js, storage.js, sample-data.js)**  
**Project Status: Production-ready with comprehensive documentation (README.md, PROJECT\_SUMMARY.md, plan.md)**  
**Founders: Priyanka Joshi (Co-Founder & Director), Ayush Anand (Co-Founder)**  
**Advisory: Dr. Chetan Arora (IIT Delhi)**

**🏫 DL Course Platform \- Educational Website with Admin Panel**  
**🔗 Repository: https://github.com/Suraj-creation/DL\_course-Shabbeer.Basha**  
**🌐 Domain: Education Technology / Learning Management System**

**📝 Description:**  
**A comprehensive full-stack educational platform designed for course instructors to create, manage, and publish course content with an intuitive admin dashboard. Built for Deep Learning courses, this LMS enables professors to manage all aspects of their courses including lectures, assignments, teaching assistants, tutorials, prerequisites, exams, and resources through a centralized admin panel. The platform features real-time updates where changes reflect immediately on the public-facing website, providing seamless content management for academic courses.**

**✨ Key Features:**  
**\- Admin Panel \- Complete course management dashboard with 8 manager modules**  
**\- Course Management \- Create and manage course information, metadata, and settings**  
**\- Lecture Management \- Upload lectures with PDF/PPT slides, YouTube video links, and reading materials**  
**\- Assignment System \- Create assignments with due dates, templates, grading status, and file uploads**  
**\- Teaching Assistants Module \- Manage TA information, contact details, and office hours**  
**\- Tutorials & Prerequisites \- Add supplementary materials and define course prerequisites**  
**\- Exam Management \- Schedule and manage exam information**  
**\- Resource Library \- Organize external resources by category (Books, Tools, Papers, etc.)**  
**\- File Upload System \- Drag-and-drop interface supporting PDFs, PPTs, documents (10MB limit)**  
**\- Real-time Updates \- Changes reflect immediately on the public website**  
**\- Public Website \- Clean, responsive interface for students with 8 public pages**  
**\- JWT Authentication \- Secure admin login with password hashing (bcryptjs)**  
**\- RESTful API \- Complete CRUD operations for all resources**  
**\- Responsive Design \- Mobile, tablet, and desktop breakpoints**

**💻 Tech Stack:**  
**\- Backend: Node.js, Express.js \- RESTful API with modular routing**  
**\- Database: MongoDB, Mongoose \- Data persistence with schema validation**  
**\- Frontend: React.js, React Router \- Component-based UI with client-side routing**  
**\- Authentication: JWT (jsonwebtoken) \- Token-based authentication**  
**\- Password Security: bcryptjs \- Password hashing and salting**  
**\- File Handling: Multer \- File upload middleware**  
**\- HTTP Client: Axios \- API communication**  
**\- Styling: TailwindCSS (to be added), React Icons \- Modern responsive design**  
**\- Languages: JavaScript (92.8%), CSS (6.4%), HTML (0.8%)**

**🛠️ Tools & Platforms:**  
**\- GitHub \- Version control (9 commits, updated 5 days ago)**  
**\- Vercel \- Deployment platform (58 deployments)**  
**\- MongoDB Atlas \- Cloud database hosting**  
**\- VS Code \- Development environment**  
**\- npm \- Package management**  
**\- PowerShell scripts \- Automated manager generation**

**🚀 Deployment:**  
**58 Vercel deployments with production hosting**  
**Local development: npm run dev (concurrent frontend and backend)**  
**Backend API: Port 5000, Frontend: Port 3000**  
**MongoDB connection: Local or MongoDB Atlas**  
**Environment variables: .env.example provided for configuration**  
**Default Admin Credentials: admin@dlcourse.com / admin123**  
**File uploads stored in ./uploads directory**  
**Documentation: 10+ comprehensive markdown files (GETTING\_STARTED, MONGODB\_SETUP, IMPLEMENTATION\_GUIDE, etc.)**  
**Project Status: Production-ready, based on CS6910 and ML Basics course website analysis**

🎲 Snake and Ladder Game \- AI-Powered Interactive Board Game

**📌 Snake and Ladder Game**  
🌐 Domain: Game Development / AI Integration  
🔗 Link: https://github.com/Suraj-creation/Snake-and-Ladder-game

🧠 Description:  
An AI-powered interactive Snake and Ladder board game built with Google Gemini AI Studio. This project represents a modern take on the classic board game, integrating artificial intelligence to enhance gameplay experience. The application was created as a personal project dedicated to family members, demonstrating how traditional games can be revitalized through AI integration. The game features smooth animations, responsive design, and intelligent game mechanics powered by Gemini API, making it an engaging digital recreation of the timeless board game.

🛠️ Tech Stack:  
\- TypeScript (91.7%) \- Primary development language  
\- HTML (8.3%)  
\- React \- UI framework (App.tsx component structure)  
\- Vite \- Build tool and development server  
\- Google Gemini AI \- AI integration for game logic

🔧 Tools & Platforms:  
\- Google AI Studio \- AI-powered development platform  
\- Node.js \- Runtime environment  
\- npm \- Package management  
\- TypeScript compiler \- Type checking  
\- Vite \- Modern build tooling

🚀 Deployment:  
\- Production URL: https://snake-and-ladder-game-ten.vercel.app/  
\- Platform: Vercel (2 deployments)  
\- Local development: npm run dev (concurrent frontend)  
\- Environment configuration: .env.local with GEMINI\_API\_KEY

**📌 Machine Learning \- EEG Alzheimer's Disease Classifier**  
🌐 Domain: Healthcare AI / Biomedical Engineering / Neuroscience  
🔗 Link: https://github.com/Suraj-creation/Machine\_learning

🧠 Description:  
A state-of-the-art machine learning platform for automated classification of Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal (CN) individuals using resting-state EEG biomarkers. This comprehensive system implements a full ML pipeline including advanced signal processing, feature engineering of 438 biomarkers, and ensemble learning to achieve clinically significant accuracy in distinguishing neurodegenerative diseases. The project leverages the OpenNeuro ds004504 dataset (88 subjects: 36 AD, 23 FTD, 29 CN) and provides an interactive Streamlit web application for real-time EEG analysis. Features include multi-channel EEG signal processing, Power Spectral Density analysis using Welch's method, non-linear dynamics metrics (entropy, fractal dimension), and hierarchical classification strategies achieving 72% accuracy for dementia screening and 77.8% recall for AD detection.

🛠️ Tech Stack:  
\- Python 3.11+ (25.1%) \- Core development language  
\- Jupyter Notebook (58.5%) \- Research and analysis pipeline  
\- TypeScript (15.8%) \- Interactive blog component  
\- Streamlit 1.28+ \- Web application framework  
\- MNE-Python 1.5+ \- EEG signal processing and visualization  
\- LightGBM 4.0+ & XGBoost 2.0+ \- Gradient boosting ensemble models  
\- NumPy, SciPy, Pandas \- Scientific computing  
\- Plotly, Matplotlib, Seaborn \- Data visualization

🔧 Tools & Platforms:  
\- OpenNeuro \- Dataset repository (ds004504 v1.0.8)  
\- BIDS \- Brain Imaging Data Structure standard  
\- Joblib \- Model persistence and serialization  
\- Pytest \- Unit and integration testing  
\- Docker \- Containerized deployment  
\- Git LFS \- Large file storage for EEG datasets  
\- ReportLab \- PDF report generation  
\- Nihon Kohden EEG 2100 \- Clinical EEG acquisition system

🚀 Deployment:  
\- Production URL: https://machine-learning-murex-two.vercel.app/  
\- Live Demo (Streamlit): https://machine-learning-suraj-creation.streamlit.app/  
\- Platform: Vercel (14 deployments)  
\- Dataset: 2.75 GB preprocessed EEG data with ASR and ICA cleaning  
\- Local development: Streamlit run with Docker Compose support  
\- Features: 438 biomarkers, 4,400+ augmented epochs, 50x sample expansion

**📌 ChatGPT Clone (Gemini Chat UI)**  
🌐 Domain: AI / Web Development / Conversational AI  
🔗 Link: https://github.com/Suraj-creation/chatgpt\_clone

🧠 Description:  
A production-quality web application that replicates the ChatGPT interface using Google's Gemini API. This comprehensive chat application demonstrates full-stack development skills with real-time streaming responses, conversation management, and an elegant dark-themed UI inspired by ChatGPT. Features include token-by-token response streaming from Gemini API, local conversation persistence using browser localStorage, model selection between Gemini 1.5 Flash/Pro and legacy models, custom system instructions for AI behavior customization, and full markdown rendering with syntax highlighting for code blocks. The application provides seamless keyboard shortcuts, responsive design for all devices, and graceful error recovery with user-friendly messages.

🛠️ Tech Stack:  
\- TypeScript (91.0%) \- Primary development language for type safety  
\- Next.js 14.2.0 \- React framework with App Router  
\- React 18.3.0 \- UI library for component architecture  
\- CSS (5.3%) \- Custom styling  
\- JavaScript (1.8%) \- Configuration files  
\- Tailwind CSS 3.4.0 \- Utility-first CSS framework  
\- react-markdown 9.0.1 \- Markdown rendering engine  
\- react-syntax-highlighter 15.5.0 \- Code syntax highlighting  
\- lucide-react 0.344.0 \- Icon library

🔧 Tools & Platforms:  
\- Google Gemini AI API \- Large language model integration  
\- localStorage \- Browser-based conversation persistence  
\- Server-Sent Events (SSE) \- Real-time streaming responses  
\- Next.js API Routes \- Backend API endpoints  
\- PostCSS \- CSS processing  
\- UUID \- Unique ID generation for messages

🚀 Deployment:  
\- Production URL: https://chatgpt-clone-taupe-one.vercel.app/  
\- Platform: Vercel (20 deployments)  
\- Node.js version: 20.x  
\- Environment: Next.js 14 with streaming support  
\- Features: Real-time chat, conversation history, model switching, dark theme

**📌 Gemini CBSE Classroom (Important\_files)**  
🌐 Domain: EdTech / AI Education / Full-Stack Development  
🔗 Link: https://github.com/Suraj-creation/Important\_files

🧠 Description:  
An AI-powered educational platform designed for CBSE curriculum learning, combining FastAPI backend with a React/Material-UI frontend. This full-stack application integrates Google's Gemini API to provide intelligent document analysis and interactive learning experiences. The system features PDF upload and rendering capabilities using PDF.js with canvas and text layer support for accurate text selection, page-by-page PDF viewing with navigation controls, chat-based interaction with educational content, and content expansion functionality for deeper learning. The platform supports multipart file uploads with base64 JSON fallback, RESTful API design with dedicated endpoints for file management and AI interactions, and seamless integration between Python backend and TypeScript frontend.

🛠️ Tech Stack:  
\- TypeScript (63.9%) \- Frontend development and type safety  
\- Python (32.6%) \- Backend API and Gemini integration  
\- CSS (3.2%) \- Custom styling and Material-UI theming  
\- HTML (0.3%) \- Base markup structure  
\- FastAPI \- High-performance Python web framework  
\- React 18+ \- Component-based UI library  
\- Material-UI (MUI) \- React component library  
\- Vite \- Modern build tool and dev server  
\- PDF.js \- Client-side PDF rendering

🔧 Tools & Platforms:  
\- Google Gemini API \- AI-powered document analysis and chat  
\- Uvicorn \- ASGI server for FastAPI  
\- Node.js 18+ \- JavaScript runtime  
\- Python 3.10+ \- Backend runtime  
\- npm \- Package management  
\- PowerShell \- Windows scripting for setup

🚀 Deployment:  
\- Backend: FastAPI server on port 8000 (http://127.0.0.1:8000)  
\- Frontend: Vite dev server with hot module replacement  
\- API Endpoints: /api/upload, /api/files, /api/file/{id}/page/{num}, /api/expand, /api/chat  
\- Features: PDF viewer, text selection, AI chat, document expansion

**📌 Live Classroom \- AI-Powered Visual Learning (ExplainBoard)**  
🌐 Domain: EdTech / AI Education / Interactive Whiteboard  
🔗 Link: https://github.com/Suraj-creation/Live\_Classroom-powered\_by\_AI

🧠 Description:  
An AI-powered visual learning whiteboard application that brings the interactive classroom experience online with intelligent explanations and real-time collaboration. ExplainBoard combines Google's Gemini AI with a classroom-inspired chalkboard interface to create an immersive educational environment. The application features dual modes (Live and Explain) for different teaching scenarios, real-time AI-generated explanations of complex concepts, classroom chalkboard aesthetic with authentic dark theme styling, and interactive whiteboard functionality for visual demonstrations. Built using Google AI Studio template, the platform provides seamless integration with Gemini API for natural language processing and concept explanations, component-based architecture for modular educational features, and responsive design for cross-device learning experiences.

🛠️ Tech Stack:  
\- TypeScript (91.7%) \- Type-safe development  
\- CSS (5.1%) \- Custom chalkboard theme styling  
\- HTML (2.4%) \- Semantic markup  
\- JavaScript (0.8%) \- Configuration  
\- React \- Component-based UI framework  
\- Vite \- Lightning-fast build tool  
\- Tailwind CSS \- Utility-first styling framework  
\- PostCSS \- CSS processing and optimization

🔧 Tools & Platforms:  
\- Google AI Studio \- AI application development platform  
\- Google Gemini API \- Large language model for explanations  
\- Node.js \- JavaScript runtime environment  
\- Vercel \- Deployment and hosting platform  
\- npm \- Package manager

🚀 Deployment:  
\- Production URL: https://live-classroom-powered-by-ai.vercel.app/  
\- Platform: Vercel (4 deployments)  
\- AI Studio integration: https://ai.studio/apps/temp/1  
\- Local development: npm run dev with Vite HMR  
\- Environment: Node.js with GEMINI\_API\_KEY configuration  
\- Features: Live teaching mode, AI explanations, chalkboard UI, interactive whiteboard

**📌 Finance Portfolio \- Enhanced Dashboard v2.0**  
🌐 Domain: FinTech / Portfolio Management / Data Visualization  
🔗 Link: https://github.com/Suraj-creation/Finance\_Portfolio\_

🧠 Description:  
An enterprise-grade Excel-themed interactive portfolio analysis dashboard implementing Modern Portfolio Theory for optimal asset allocation. This comprehensive financial analysis platform features a 100% functional Excel UI with 7 ribbon tabs and 50+ working buttons, interactive Plotly.js-powered visualizations including efficient frontier, weight evolution, and performance comparison charts, and SLSQP optimization achieving 541.9% Sharpe ratio improvement. The system analyzes 5 Indian equity securities over 258 trading days, delivering optimal weights (58.11% MARUTI, 21.51% M\&M, 20.38% HYUNDAI) and transforming \-2.35% equal-weight returns into \+29.05% annual returns with reduced volatility. Features include 20+ keyboard shortcuts, auto-save every 30 seconds, global search, export to PDF, and comprehensive documentation.

🛠️ Tech Stack:  
\- HTML (99.8%) \- Structure and semantic markup  
\- CSS (0.2%) \- Styling, animations, Excel theme  
\- JavaScript ES6+ \- Interactivity and data management  
\- Plotly.js 3.1.0 \- Interactive chart visualizations  
\- PapaParse 5.4.1 \- CSV data parsing  
\- Font Awesome 6.0.0 \- Icon library  
\- Animate.css 4.1.1 \- CSS animations

🔧 Tools & Platforms:  
\- Yahoo Finance API \- Historical price data source  
\- localStorage \- State persistence and auto-save  
\- Modern Portfolio Theory \- Sharpe ratio maximization  
\- SLSQP Algorithm \- Sequential Least Squares optimization  
\- Python SciPy \- Backend optimization calculations  
\- Microsoft Excel \- Data verification and export

🚀 Deployment:  
\- Platform: Vercel (3 deployments)  
\- Load time: \<2 seconds  
\- Browser support: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+  
\- Features: Auto-save, PDF export, responsive design, keyboard shortcuts  
\- Performance: 60fps animations, \~50MB memory, localStorage persistence

**📌 AI-Powered Note-Making Mobile App**  
🌐 Domain: Mobile Development / AI Productivity / Note-Taking  
🔗 Link: https://github.com/Suraj-creation/AI\_powered\_notemaking\_mobile\_app

🧠 Description:  
An intelligent mobile note-taking application powered by Google's Gemini AI, built using Google AI Studio template. This modern productivity app combines traditional note-taking with AI capabilities to enhance the user experience through context-aware suggestions, smart organization, and AI-assisted content generation. The application features a component-based React architecture with TypeScript for type safety, custom hooks for reusable logic and state management, context API for global state handling, and service layer architecture for API integrations. Built with mobile-first responsive design principles, the app provides seamless note creation, editing, and management with AI-powered enhancements that help users capture and organize their thoughts more effectively.

🛠️ Tech Stack:  
\- TypeScript (97.6%) \- Type-safe development  
\- HTML (2.4%) \- Mobile-optimized markup  
\- React \- Component-based UI framework  
\- React Context API \- State management  
\- Vite \- Fast build tool and dev server  
\- Google Gemini AI \- AI-powered features

🔧 Tools & Platforms:  
\- Google AI Studio \- AI application development  
\- Gemini API \- Large language model integration  
\- Node.js \- JavaScript runtime  
\- npm \- Package manager  
\- Vercel \- Deployment platform  
\- AI Studio integration: https://ai.studio/apps/drive/1EHMFbWpfnVZ5JBz7To5R\_Sx2LsZ2jJPr

🚀 Deployment:  
\- Production URL: https://ai-powered-notemaking-mobile-app.vercel.app/  
\- Platform: Vercel (1 deployment)  
\- Environment: Node.js with GEMINI\_API\_KEY  
\- Local dev: npm run dev with Vite HMR  
\- Features: AI suggestions, smart notes, context management, mobile-responsive UI

**📌 Echo Chamber Buster \- Challenge Your Reasoning**  
🌐 Domain: AI / Philosophy / Critical Thinking / EdTech  
🔗 Link: https://github.com/Suraj-creation/Challenge\_your\_Reasoning

🧠 Description:  
An AI-powered adversarial debate platform that challenges beliefs through relentless, evidence-based philosophical sparring. Echo Chamber Buster fosters critical thinking and intellectual humility by taking adversarial stances on 40+ controversial topics across 8 domains (Life & Existence, Ethics & Morality, Rights & Justice, Society & Politics, Science & Philosophy, etc.). The AI, powered by Google Gemini, never agrees with the user and systematically dismantles arguments using authoritative sources including great thinkers (Socrates, Nietzsche, MLK, Gandhi, Einstein, Freud), spiritual texts (Bhagavad Gita, Bible, Quran, Tao Te Ching), and peer-reviewed research. Features include single-file self-contained architecture with zero dependencies, ChatGPT-inspired interface with light/dark theme, exponential backoff retry logic with specific rate limit handling, session management with conversation history, and surrender detection with graceful wrap-ups. Built on the philosophy that no truth is absolute—flaws lurk in every certainty.

🛠️ Tech Stack:  
\- HTML (100.0%) \- Structure and embedded logic  
\- CSS3 \- Modern styling with theme system  
\- JavaScript ES6+ \- Client-side application logic  
\- Google Gemini API \- gemini-1.5-flash model  
\- localStorage/sessionStorage \- State persistence

🔧 Tools & Platforms:  
\- Google Gemini AI \- Large language model  
\- Google AI Studio \- API key management  
\- Prompt engineering \- Adversarial debate system  
\- CSP (Content Security Policy) \- Security hardening  
\- Browser APIs \- No external dependencies

🚀 Deployment:  
\- Production URL: https://challenge-your-reasoning.vercel.app/  
\- Platform: Vercel (4 deployments)  
\- Browser support: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+  
\- Architecture: Single HTML file, no build process  
\- Works on file:// protocol for offline use  
\- Features: 40+ topics, theme persistence, debate history, surrender detection  
──────────────────────────────────────────────────────────  
📌 Project Name: NotemakingAI (Thought Canvas)  
🌐 Domain / Field: Mobile Development, Android, AI Productivity, Note-Taking, Content Enhancement  
🔗 GitHub Repository:  
https://github.com/Suraj-creation/NotemakingAI

🧠 Project Description:  
Thought Canvas is an AI-powered note-taking Android application that transforms raw thoughts into polished, structured content using Google's Gemini AI. The app implements a dual content model that maintains both raw thoughts and AI-polished versions, allowing users to track the evolution of their ideas. It features smart AI enhancement through Gemini that transforms messy notes into structured content, comprehensive version history with human vs AI attribution, and automatic note enhancement through background sync every 12 hours. The application includes AI-suggested tags for better organization, automatic extraction of actionable tasks, and operates with an offline-first architecture ensuring full functionality without internet connection.

Built with modern Android development practices, the app follows Clean Architecture with clear separation of concerns, implements MVVM pattern with StateFlow for reactive UI, uses Jetpack Compose for modern declarative UI, Room Database for local persistence, Retrofit for API communication, and WorkManager for background processing. The project demonstrates advanced Android development skills including proper architecture patterns, modern UI frameworks, background task management, and AI API integration. The app provides a seamless user experience with automatic background enhancements, smart organization features, and offline-first design.

🛠️ Tech Stack:  
\- Kotlin (100.0%) \- Primary programming language  
\- Jetpack Compose \- Modern declarative UI framework  
\- Material Design 3 \- UI design system  
\- Room Database (SQLite) \- Local data persistence  
\- Retrofit \- REST API client for network calls  
\- Moshi \- JSON parsing library  
\- WorkManager \- Background task scheduling  
\- Navigation Compose \- Navigation framework  
\- Google Gemini AI \- AI content enhancement  
\- StateFlow \- Reactive state management  
\- Coroutines \- Asynchronous programming

🔧 Tools & Platforms:  
\- Google Gemini AI API \- Large language model for content enhancement  
\- Android Studio Hedgehog+ \- IDE  
\- Android SDK 26+ (target SDK 34\) \- Platform APIs  
\- Gradle with Kotlin DSL \- Build system  
\- Git \- Version control  
\- MVVM Architecture Pattern \- Presentation layer  
\- Clean Architecture \- Domain/Data/UI separation

🚀 Deployment:  
\- Platform: Android mobile application (APK)  
\- Not deployed to Play Store (development phase)  
\- Local installation: ./gradlew assembleDebug  
\- Requires GEMINI\_API\_KEY in local.properties  
\- Target: Android 8.0+ (API 26+)  
\- Architecture: Single HTML file, no build process  
\- Features: Dual content model, version history, background sync (12h), AI tagging, action items, offline-first

──────────────────────────────────────────────────────────  
📌 Project Name: DL Course Platform (Educational Website)  
🌐 Domain / Field: Educational Technology, Course Management System, E-Learning, Full-Stack Web Development  
🔗 GitHub Repository:  
https://github.com/Suraj-creation/DL\_course-Shabbeer.Basha

🧠 Project Description:  
A comprehensive full-stack educational platform designed for course instructors to create, manage, and publish course content with an intuitive admin dashboard. The platform serves as a complete Learning Management System (LMS) enabling educators to efficiently manage all aspects of their courses. The system consists of two main components: an Admin Panel for instructors to manage content and a Public Website for students to access course materials.

The Admin Panel provides robust course management capabilities including course creation with metadata and settings, lecture management with slide uploads (PDF/PPT), video integration (YouTube links), and reading materials. It features a comprehensive assignment system with due dates, templates, and grading status tracking. Instructors can manage teaching assistants with contact details and office hours, add supplementary tutorials with resources and practice problems, define course prerequisites with external resource links, schedule and manage exams, and organize external resources by category (Books, Tools, Papers, etc.). The system supports file uploads with drag-and-drop interface and provides real-time updates that reflect immediately on the public website.

The Public Website offers a clean, responsive design with easy navigation structure showcasing all course components. It features a hero section with course highlights, organized lecture lists with all materials, assignment tracking with status indicators, TA directory with contact information and office hours, and a categorized resource library. Built with Node.js/Express backend and React frontend, the platform implements JWT-based authentication, MongoDB for data persistence, and supports file uploads up to 10MB. The project is based on analysis of CS6910 and ML Basics course websites and demonstrates advanced full-stack development skills.

🛠️ Tech Stack:  
\- Node.js \+ Express.js \- RESTful API backend  
\- MongoDB \+ Mongoose \- NoSQL database and ODM  
\- React.js \- Frontend UI framework  
\- React Router \- Client-side navigation  
\- Axios \- HTTP client for API calls  
\- JWT (JSON Web Tokens) \- Authentication system  
\- Multer \- File upload middleware  
\- bcryptjs \- Password hashing and security  
\- React Icons \- Icon library  
\- JavaScript (92.8%) \- Primary language  
\- CSS (6.4%) \- Styling  
\- HTML (0.8%) \- Markup

🔧 Tools & Platforms:  
\- MongoDB Atlas \- Cloud database hosting  
\- Vercel \- Deployment platform (58 deployments)  
\- npm \- Package manager  
\- Node.js runtime \- Server environment  
\- File Upload System \- Multer-based with 10MB limit  
\- RESTful API \- Standard HTTP methods and routes

🚀 Deployment:  
\- Production URL: Multiple Vercel deployments (58 total)  
\- Platform: Vercel with production environment  
\- Backend: Node.js/Express on port 5000  
\- Frontend: React build served at port 3000  
\- Database: MongoDB (local or Atlas)  
\- Admin Panel: /admin route with JWT authentication  
\- Default Credentials: admin@dlcourse.com / admin123  
\- Features: 8 admin managers (Courses, Lectures, Assignments, TAs, Tutorials, Prerequisites, Exams, Resources), 8 public pages, file uploads, real-time updates, responsive design

──────────────────────────────────────────────────────────  
📌 Project Name: Snake and Ladder Game  
🌐 Domain / Field: Game Development, Web-Based Gaming, Educational Games, Interactive Applications  
🔗 GitHub Repository:  
https://github.com/Suraj-creation/Snake-and-Ladder-game

🧠 Project Description:  
A web-based implementation of the classic Snake and Ladder board game, dedicated to the developer's nephews Reesu and Reetu. This interactive game application was built using Google's AI Studio repository template, showcasing modern web development practices for creating engaging browser-based games. The project demonstrates the integration of game logic, smooth animations, and polished user interface design to deliver an enjoyable gaming experience.

The game features enhanced animations and styling for smooth transitions, providing a visually appealing and responsive gameplay experience. Built with React and TypeScript, the application follows component-based architecture with separate modules for game constants, types, utilities, and UI components. The project uses Vite as the build tool for fast development and optimized production builds. The game logic handles player movements, snake and ladder mechanics, and turn-based gameplay, while the UI provides an interactive board with visual feedback for game events. This project represents a personal touch, creating educational and entertaining content for younger family members.

🛠️ Tech Stack:  
\- TypeScript (91.7%) \- Type-safe programming language  
\- HTML (8.3%) \- Markup and structure  
\- React \- UI component framework  
\- Vite \- Modern build tool and dev server  
\- TSX/JSX \- Component markup syntax  
\- CSS \- Styling with animations

🔧 Tools & Platforms:  
\- Google AI Studio \- Project template and scaffolding  
\- AI Studio Integration: https://ai.studio/apps/drive/17Q7-FDdkLie7JlnYBk0NJx0Bic9uAKeN  
\- Vercel \- Deployment platform  
\- Node.js \- Runtime environment  
\- npm \- Package manager  
\- Gemini API \- AI integration (requires GEMINI\_API\_KEY)

🚀 Deployment:  
\- Production URL: https://snake-and-ladder-game-ten.vercel.app/  
\- Platform: Vercel (2 deployments)  
\- Environment: Node.js with Gemini API key  
\- Local dev: npm run dev  
\- Features: Interactive board, smooth animations, enhanced styling, turn-based gameplay, snake/ladder mechanics  
\- Personal dedication: Created for nephews Reesu and Reetu

──────────────────────────────────────────────────────────  
📌 Project Name: Healthcare AI Assistant v2.0  
🌐 Domain / Field: Healthcare Technology, AI-Powered Diagnostics, Medical Informatics, Preventive Healthcare  
🔗 GitHub Repository:  
https://github.com/Suraj-creation/Healthcare\_Prediction

🧠 Project Description:  
A cutting-edge, AI-powered healthcare diagnostic assistant that combines machine learning with Google Gemini 2.5 Flash AI to provide intelligent disease predictions, personalized health recommendations, and real-time medical insights based on user symptoms. This comprehensive web application serves as an educational tool for health awareness and preliminary symptom assessment, featuring 41 disease models trained on medical datasets and a database of 132 symptoms with severity weighting (1-7 scale).

The system implements a privacy-first, local-first architecture where all user data remains stored in the browser's localStorage with no backend server required. The Smart Symptom Checker offers multi-modal input including text search with autocomplete, interactive SVG body map with clickable regions, and voice recognition via Web Speech API. The ML prediction engine analyzes symptoms using custom algorithms to identify top 5 probable conditions with percentage-based confidence scoring and color-coded severity assessment (Low/Medium/High/Critical).

Google Gemini AI integration provides natural language Q\&A capabilities, contextual understanding of user symptoms, personalized health insights, and access to vast medical literature. The application features interactive ECharts.js visualizations for symptom analysis, recovery timeline tracking, and health score dashboards. Users receive comprehensive personalized recommendations across 6 categories: disease overview, medications (drugs/dosage/timing), custom diet plans, tailored exercise routines, safety precautions, and recovery timeline with milestones. The modern glassmorphism UI with Anime.js animations ensures smooth micro-interactions and responsive design across all devices (mobile/tablet/desktop). Privacy features include data export/import as JSON, one-click data deletion, and WCAG 2.1 AA accessibility compliance.

🛠️ Tech Stack:  
\- HTML5 (86.6%) \- Semantic markup with accessibility  
\- JavaScript ES6+ (13.4%) \- Modern async/await, modules, classes  
\- Tailwind CSS \- Utility-first CSS framework  
\- Google Gemini 2.5 Flash \- Latest AI model for healthcare insights  
\- ECharts.js 5.x \- Interactive charts and data visualization  
\- Anime.js 3.x \- Smooth animations and micro-interactions  
\- Typed.js 2.x \- Typewriter effects for AI responses  
\- Splitting.js 1.x \- Advanced text animations  
\- Web Speech API \- Voice recognition for symptom input  
\- LocalStorage API \- Client-side data persistence  
\- Fetch API \- Modern HTTP requests to AI services

🔧 Tools & Platforms:  
\- Google Gemini AI API \- Natural language AI responses with 32K token context window  
\- Vercel \- Deployment platform (7 deployments)  
\- Python HTTP server \- Local development  
\- VS Code \- Development environment  
\- Custom ML Models \- Trained disease prediction algorithms  
\- Medical Knowledge: WHO, CDC, PubMed, Mayo Clinic databases

🚀 Deployment:  
\- Production URL: https://healthcare-prediction.vercel.app/  
\- Platform: Vercel with production environment  
\- Version: 2.0 (October 2025\)  
\- Local setup: python \-m http.server 8000  
\- Browser support: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+  
\- Performance: \<2 second AI responses, optimized rendering  
\- Features: 41 diseases, 132 symptoms, top 5 predictions, confidence scoring, severity classification, AI Q\&A, voice input, body map interaction, personalized recommendations, health profile management, data export/import, WCAG 2.1 AA accessible, responsive design, glassmorphism UI, privacy-first architecture  
\- Medical Disclaimer: Educational tool only, not for diagnosis/treatment decisions

──────────────────────────────────────────────────────────  
📌 Project Name: Image Captioning & Segmentation  
🌐 Domain / Field: Computer Vision, Deep Learning, Image Processing, Neural Networks, AI Research  
🔗 GitHub Repository:  
https://github.com/Suraj-creation/Image\_captioning\_-\_Segmentation

🧠 Project Description:  
A production-quality Streamlit web application that combines image captioning and image segmentation into a unified deep learning pipeline using the COCO 2014 dataset with state-of-the-art models. This comprehensive computer vision project employs CNNs, LSTMs, Transformers, and Mask R-CNN for accurate object labeling and automatic text description generation from images.

The application provides dual functionality supporting both image captioning (generating natural language descriptions of images) and image segmentation (identifying and outlining objects at pixel level). Multiple model architectures are supported including ResNet50+LSTM and InceptionV3+Transformer for captioning, plus Mask R-CNN (instance segmentation), DeepLabV3+ (semantic segmentation), and U-Net (semantic segmentation adapted from medical imaging) for segmentation tasks.

Key features include a combined pipeline that synchronizes caption and segmentation with object linking, batch processing capabilities for multiple images, interactive Streamlit UI with real-time visualization, Docker support for both CPU and GPU-enabled containers, comprehensive testing with CI/CD integration, and production-quality error handling. The system provides advanced controls including beam search width adjustment (1-5), maximum caption length configuration (10-30), confidence threshold tuning for instance segmentation, mask transparency control (0-1), bounding boxes and labels toggling, and class-wise legend visualization.

Users can upload images via file upload (PNG/JPG/JPEG, max 10MB), select from sample COCO 2014 dataset, or load from web URL. The application generates captions with confidence scores, token-level probabilities, and BLEU/CIDEr metrics calculation with reference captions. Segmentation results include detection details inspection and raw mask array visualization. Developer mode provides model load times monitoring, memory usage tracking, intermediate feature maps inspection, token probabilities analysis, and segmentation mask debugging.

🛠️ Tech Stack:  
\- Python (89.6%) \- Core programming language  
\- PowerShell (6.4%) \- Deployment scripts  
\- CSS (2.2%) \- Custom UI styling  
\- Streamlit \- Web framework for interactive UI  
\- PyTorch \- Deep learning framework  
\- NLTK \- Natural language processing for captioning  
\- OpenCV \- Computer vision library  
\- NumPy \- Numerical computing  
\- Pillow \- Image processing

🔧 Tools & Platforms:  
\- ResNet50 \+ LSTM \- CNN encoder with LSTM decoder for captioning  
\- InceptionV3 \+ Transformer \- InceptionV3 encoder with Transformer decoder  
\- Mask R-CNN \- Pretrained instance segmentation on COCO 2014  
\- DeepLabV3+ \- Advanced semantic segmentation  
\- U-Net \- Medical imaging architecture adapted for COCO  
\- COCO 2014 Dataset \- Benchmark dataset for training and evaluation  
\- Docker & Docker Compose \- Containerization (CPU and GPU support)  
\- GitHub Actions \- CI/CD pipeline  
\- pytest \- Testing framework with coverage reporting

🚀 Deployment:  
\- Production URL: https://image-captioning-segmentation-nu.vercel.app/  
\- Platform: Vercel (9 deployments)  
\- Streamlit Cloud compatible  
\- Docker support: CPU and GPU versions available  
\- Local development: streamlit run app.py (localhost:8501)  
\- Python version: 3.10+  
\- License: MIT  
\- Features: Dual captioning/segmentation, multiple models, combined pipeline, batch processing, real-time visualization, Docker ready, comprehensive testing, CI/CD, interactive UI, metrics calculation (BLEU/CIDEr), developer mode, export capabilities (caption.txt, mask.png, results.json, batch ZIP)  
\- Architecture: Modular pipeline with separate inference modules, model wrappers, utility modules for visualization/COCO/I/O

──────────────────────────────────────────────────────────  
📌 Project Name: Gemini Chat UI (ChatGPT Clone)  
🌐 Domain / Field: AI / Web Development / Conversational AI  
🔗 GitHub Repository:  
https://github.com/Suraj-creation/chatgpt\_clone

🧠 Project Description:  
A production-quality web application that replicates the ChatGPT interface using Google's Gemini API, built with Next.js 14, TypeScript, and Tailwind CSS. This modern conversational AI application provides real-time streaming responses, comprehensive conversation management, and a beautiful dark theme interface inspired by ChatGPT.

Core functionality includes real-time token-by-token response streaming from Gemini API, complete conversation management (create, save, load, delete), model selection between Gemini 1.5 Flash/Pro and legacy models, local persistence with all conversations saved to browser localStorage, and a beautiful dark theme interface. Advanced features include customizable system instructions for AI behavior, full markdown rendering with syntax highlighting, one-click code copy functionality, keyboard shortcuts (Enter to send, Shift+Enter for newline), responsive design for all devices, and graceful error recovery with user-friendly messages.

The application architecture follows Next.js App Router pattern with a /api/chat route handling Gemini API integration, component-based structure with Sidebar/ConversationList/ChatHeader/MessageList/MessageBubble/ChatInput/ModelSelector/SystemPromptEditor/LoadingIndicator, global state management through React Context, localStorage utilities for persistence, and TypeScript definitions for type safety. Users can switch between multiple Gemini models, customize AI behavior with system prompts, manage past conversations in the sidebar, and enjoy markdown rendering with syntax-highlighted code blocks.

🛠️ Tech Stack:  
\- TypeScript (91.0%) \- Type-safe programming  
\- CSS (5.3%) \- Custom styling  
\- JavaScript (1.8%) \- Configuration  
\- Next.js 14.2.0 \- React framework with App Router  
\- React 18.3.0 \- UI library  
\- Tailwind CSS 3.4.0 \- Utility-first styling  
\- react-markdown 9.0.1 \- Markdown rendering  
\- react-syntax-highlighter 15.5.0 \- Code highlighting  
\- lucide-react 0.344.0 \- Icon library  
\- uuid 9.0.1 \- ID generation

🔧 Tools & Platforms:  
\- Google Gemini API \- AI language model (Flash, Pro, legacy)  
\- Vercel \- Deployment platform (20 deployments)  
\- Node.js 20.x \- Runtime environment  
\- localStorage \- Browser storage for conversations  
\- Server-Sent Events (SSE) \- Streaming responses

🚀 Deployment:  
\- Production URL: https://chatgpt-clone-taupe-one.vercel.app/  
\- Platform: Vercel with environment variables  
\- Node version: 20.x  
\- Local dev: npm run dev (localhost:3000)  
\- Build: npm run build && npm start  
\- Features: Real-time streaming, conversation management, model selection, system instructions, markdown support, code copy, keyboard shortcuts, dark theme, responsive design, error handling, localStorage persistence  
\- Environment: GEMINI\_API\_KEY (required), NEXT\_PUBLIC\_APP\_NAME (optional)

📌 Project Name: Live Classroom \- AI-Powered Visual Learning

🌐 Domain/Field: EdTech, AI-Powered Education, Interactive Learning

🔗 GitHub Repository: https://github.com/Suraj-creation/Live\_Classroom-powered\_by\_AI

🧠 Project Description:  
ExplainBoard is an AI-powered visual learning whiteboard designed for interactive classroom experiences. Built with Google AI Studio and Gemini API, it provides the fastest path from prompt to production for educational AI applications. The platform enables real-time AI-assisted teaching and learning through an interactive whiteboard interface, combining conversational AI with visual learning tools to create an engaging educational environment.

🛠️ Tech Stack:  
\- React with TypeScript  
\- Vite \- Build tool and development server  
\- Tailwind CSS 3.4.0 \- Utility-first styling  
\- react-markdown 9.0.1 \- Markdown rendering  
\- react-syntax-highlighter 15.5.0 \- Code highlighting  
\- lucide-react 0.344.0 \- Icon library  
\- uuid 9.0.1 \- ID generation  
\- Google Gemini API \- AI language model (Flash, Pro, legacy)

🔧 Tools & Platforms:  
\- Google AI Studio \- AI application platform  
\- Vercel \- Deployment platform (20 deployments)  
\- Node.js 20.x \- Runtime environment  
\- localStorage \- Browser storage for conversations  
\- Server-Sent Events (SSE) \- Streaming responses

🚀 Deployment:  
\- Production URL: https://chatgpt-clone-taupe-one.vercel.app/  
\- Platform: Vercel with environment variables  
\- Node version: 20.x  
\- Local dev: npm run dev (localhost:3000)  
\- Build: npm run build && npm start  
\- Features: Real-time streaming, conversation management, model selection, system instructions, markdown support, code copy, keyboard shortcuts, dark theme, responsive design, error handling, localStorage persistence  
\- Environment: GEMINI\_API\_KEY (required), NEXT\_PUBLIC\_APP\_NAME (optional)

📌 Project Name: Portfolio Finance Optimizer (MPT Dashboard)

🌐 Domain/Field: Financial Technology, Portfolio Management, Quantitative Finance

🔗 GitHub Repository: https://github.com/Suraj-creation/Portfolio\_finance\_Optimal

🧠 Project Description:  
An Excel-themed financial portfolio dashboard implementing Modern Portfolio Theory (MPT) for optimal asset allocation. Using real historical data from the National Stock Exchange of India (2020-2024), the platform performs SLSQP optimization to maximize the Sharpe ratio across 8 NSE assets. The application achieved a 15.6% improvement in Sharpe ratio (from 1.192 to 1.377) by reallocating weights based on risk-adjusted returns. Features AI-powered insights via Google Gemini 1.5 Pro, interactive visualizations of efficient frontier with Monte Carlo simulations, correlation matrices, and comprehensive risk analytics. The research includes a 2,243-line Jupyter notebook with complete MPT implementation and data extraction from an 11-sheet Excel workbook containing 1,237 price observations.

🛠️ Tech Stack:  
\- Python 3.14+ with pandas 2.3.3, numpy 2.3.4  
\- scipy.optimize \- SLSQP algorithm for portfolio optimization  
\- yfinance \- Yahoo Finance API for NSE data  
\- Frontend: HTML5, TailwindCSS (Excel theme), JavaScript ES6+  
\- Plotly.js v3.0.3 \- Interactive charts (heatmaps, scatter)  
\- ECharts v5.4.3 \- Additional visualizations  
\- Font Awesome v6.4.0 \- Icons  
\- Google Gemini 1.5 Pro API \- AI insights  
\- matplotlib/seaborn \- Research visualizations

🔧 Tools & Platforms:  
\- Jupyter Notebook \- Research and analysis (2243 lines)  
\- Excel (11 sheets) \- Source data workbook  
\- Vercel \- Deployment platform  
\- Yahoo Finance API \- Historical NSE stock data  
\- Google Gemini API \- AI-powered portfolio analysis

🚀 Deployment:  
\- Production URL: https://portfolio-finance-optimal.vercel.app  
\- Local: Open index.html in browser (no build required)  
\- Data: 8 NSE assets (M\&M, Adani Power, Vedanta, Bharti Airtel, NTPC, Tata Steel, Adani Ports, SBIN)  
\- Optimization: Max Sharpe ratio 1.377 (15.6% improvement over equal-weighted)  
\- Key Results: Expected return 42.58% (+5.37%), Volatility 26.19%, Risk-free rate 6.5%  
\- Features: Efficient frontier (30 points), Monte Carlo (100 portfolios), correlation matrices, AI confidence scores (92% optimization, 88% risk assessment), interactive visualizations, Excel/PDF export, sector diversification (Auto, Telecom, Metals, Power)  
\- Pages: index.html (portfolio overview), analytics.html (risk analysis), optimization.html (efficient frontier), insights.html (AI analysis)  
