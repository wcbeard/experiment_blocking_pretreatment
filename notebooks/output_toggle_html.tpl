{%- extends 'hide.tpl' -%}
<!-- 'basic.tpl' -->


{%- block header -%}
{{ super() }}

<script src="//ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>
<link href='https://fonts.googleapis.com/css?family=Source+Sans+Pro' rel='stylesheet' type='text/css'>


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [["$","$"],["\\(","\\)"]]}
  });
</script>

<!-- <script type="text/javascript" src="/MathJax/MathJax.js?config=TeX-AMS_HTML-full"></script> -->
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML-full"> </script>
<style type="text/css">
  .input_hidden {
    display: none;
    //  margin-top: 5px;
  }
  .container {
    /*width:140%; //!important;*/
    width: none; //!important;
    border: none !important;
  }
  div.prompt {
    display: none;
  }
  .CodeMirror{
    /*font-family: "Consolas", sans-serif;*/
    font-size:18px;
    line-height: 30px;
  }
  pre, kbd, samp {
    /*font-family: 'Source Sans Pro', Helvetica, Arial, sans-serif;*/
    /*line-height: 30px;*/
    font-size:18px;
  }
  pre.code {
    font-family: Consolas, monospace;
    font-size: 12px;
  }*/
  /*p {
    font-size:20px;
    font-family: 'Source Sans Pro', Helvetica, Arial, sans-serif;
    line-height: 30px;
    text-align: left;
  }*/
  div.cell{
    max-width:100%;
    margin-left:auto;
    margin-right:auto;
  }
  div.text_cell_render{
    max-width:100%;
    margin-left:auto;
    margin-right:auto;
  }
  h2,h3,h4{
    text-align: left;
  }
  </style>

  <script>
  $(document).ready(function(){
    $(".output_wrapper").click(function(){
      $(this).prev('.input_hidden').slideToggle();
    });
  })
  </script>
  {%- endblock header -%}
