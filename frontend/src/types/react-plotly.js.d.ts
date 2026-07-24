declare module 'react-plotly.js' {
  import { Component } from 'react';
  export interface PlotlyComponentProps {
    data: any[];
    layout?: any;
    config?: any;
    onClick?: (event: any) => void;
    onSelected?: (event: any) => void;
    style?: React.CSSProperties;
    className?: string;
  }
  export default class Plot extends Component<PlotlyComponentProps> {}
}
