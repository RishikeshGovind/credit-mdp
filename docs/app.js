const io = new IntersectionObserver((es)=>{
    es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); }});
  }, {threshold:0.12});
  document.querySelectorAll('.fade').forEach(el=>io.observe(el));

document.querySelectorAll('.viz').forEach(function(viz){
    var tabs=viz.querySelectorAll('.viz-tab');
    var chart=viz.querySelector('.viz-chart');
    var data=viz.querySelector('.viz-data');
    tabs.forEach(function(tab){
      tab.addEventListener('click',function(){
        tabs.forEach(function(t){t.classList.remove('is-active');});
        tab.classList.add('is-active');
        var v=tab.dataset.view;
        chart.hidden=(v!=='chart'); data.hidden=(v!=='data');
      });
    });
  });
