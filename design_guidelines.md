# QuantumBoost ML Design Guidelines

## Design Approach
**System-Based Approach**: Material Design principles adapted for data science applications, emphasizing clarity, efficiency, and data visualization prominence. This is a utility-focused tool where function drives form.

## Typography System

**Font Stack**: 
- Primary: 'Inter' via Google Fonts (clean, technical readability)
- Monospace: 'JetBrains Mono' for data values, code snippets, metrics

**Hierarchy**:
- Page titles: text-3xl font-semibold
- Section headers: text-xl font-semibold
- Subsection headers: text-lg font-medium
- Body text: text-base
- Data labels: text-sm font-medium
- Metric values: text-2xl font-bold (monospace)
- Table headers: text-xs font-semibold uppercase tracking-wide

## Layout System

**Spacing Primitives**: Use Tailwind units of 4, 6, 8, and 12 consistently
- Component padding: p-6 or p-8
- Section margins: mb-8 or mb-12
- Form field spacing: space-y-4
- Grid gaps: gap-6

**Container Structure**:
- Max width: max-w-7xl mx-auto
- Page padding: px-6 lg:px-8
- Content cards: Contained in white cards with subtle shadow

## Core Components

### Navigation
- Top bar: Fixed header with app title, current page indicator
- Minimal design: No complex menus, simple breadcrumb-style navigation showing current step

### Data Upload Section (Index Page)
- Centered card layout (max-w-2xl)
- Drag-and-drop zone with clear upload affordance
- File type badges showing accepted formats (.csv, .xlsx)
- Model selection: Radio buttons with clear labels and brief descriptions
- Hyperparameter controls: Grouped in collapsible sections (not all visible at once)
- Primary CTA: Large, prominent button

### Data Inspection Dashboard
- Grid layout for inspection metrics: 2-column on desktop, 1-column mobile
- Each metric card: White background, subtle border, p-6
- Data tables: Striped rows, sticky headers, fixed-width monospace for numbers
- Statistical cards: Large metric value centered, label below
- Charts: Full-width within cards, generous padding around visualizations
- Correlation heatmap: Square aspect ratio, centered
- Preprocessing controls: Sidebar or top panel with form inputs grouped logically

### Training Results Layout
- Hero metrics section: 3-column grid showing MAE, RMSE, MAPE as large cards
- Comparison table: Full-width, alternating row colors, sticky header
- Visualization section: 2-column grid for plots (loss curves, optimizer trajectory)
- Forecast chart: Full-width, prominent placement
- Download PDF button: Fixed bottom-right or top-right, always accessible

### Forms & Controls
- Input fields: border-gray-300, focus:border-blue-500, rounded-md
- Dropdowns: Native select styling enhanced with Tailwind
- Checkboxes/Radio: Larger click targets (h-4 w-4 minimum)
- Labels: Positioned above inputs, font-medium
- Help text: text-sm text-gray-500 below inputs

### Data Visualization Cards
- Plots embedded in cards with titles and descriptions
- Matplotlib figures: White background, transparent where appropriate
- Chart containers: aspect-video or aspect-square depending on content
- Caption text below each visualization explaining what it shows

### Tables
- Header row: bg-gray-50 with dark text
- Striped rows: even rows with subtle bg-gray-50
- Numeric columns: Right-aligned, monospace font
- Borders: Minimal - only outer border and header separator
- Responsive: Horizontal scroll on mobile if needed

## Page-Specific Layouts

### Index Page
- Clean, centered single-column layout
- Hero section: Brief description of QuantumBoost (text-only, concise)
- Upload card: Prominent, centered
- Sample dataset button: Secondary style, below upload
- Model selection and parameters: Revealed after file selection

### Inspect Page
- Full-width dashboard layout
- Top section: Dataset summary (shape, missing values, duplicates) in metric cards
- Middle section: Tables (head/tail) in tabbed interface
- Visualization section: Grid of charts (correlation, outliers, time-series preview)
- Bottom section: Preprocessing controls in sticky panel
- "Proceed to Training" button: Large, primary style, fixed or prominent

### Results Page
- Top: Summary metrics in hero cards
- Middle: Tabbed or accordion layout for different visualizations
- Comparison table: Prominent, full-width
- Plots: Stacked vertically with clear section headers
- Download PDF: Sticky action button

## Interaction Patterns
- Progressive disclosure: Show complexity only when needed
- Loading states: Spinner with descriptive text during training
- Success/error messages: Toast-style notifications or inline alerts
- Form validation: Inline, immediate feedback

## Data Density Strategy
- Embrace information density - this is a professional tool
- Use visual hierarchy through typography and spacing, not excessive whitespace
- Group related information in cards/sections
- Tables can be dense but must maintain scannability

## Accessibility
- Semantic HTML for all form elements
- ARIA labels for complex controls
- Sufficient color contrast for all text
- Keyboard navigation for all interactive elements
- Focus indicators on all focusable elements

## Images
No decorative images needed. All visual content is functional:
- Chart/plot images: Generated by Matplotlib, embedded in cards
- Data visualization is the primary visual content
- No hero images - this is a utility application