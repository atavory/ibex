/*
* Card Deck Slideshow (c) Dynamic Drive (www.dynamicdrive.com)
* Last updated: May 2nd, 2013
* Visit http://www.dynamicdrive.com/ for this script and 100s more.
* Requires: jQuery 1.7 or higher
*/

var carddeckslideshow = (function($){

	var defaults = {
		hingepoint: 'top left',
		fxduration: 1200,
		source: 'inline',
		activeclass: 'hinge',
		autoplay: false,
		cycles: 2,
		persist: true,
		pause: 3000,
		onunhinge: function(curindex, previndex){}
	}

	var transform3d = typeof $(document.documentElement).css('perspective') != "undefined" // test for support for 3D transform

	function getCookie(Name){ 
		var re=new RegExp(Name+"=[^;]+", "i") //construct RE to search for target name/value pair
		if (document.cookie.match(re)) //if cookie found
			return document.cookie.match(re)[0].split("=")[1] //return its value
		return null
	}

	function setCookie(name, value){
		document.cookie = name+"="+value
	}

	function carddeckslideshow(options){
		this.setting = $.extend({}, defaults, options)
		this.setting.pause += this.setting.fxduration
		var setting = this.setting
		this.curindex = ( setting.persist && getCookie(setting.id + '_lastviewedpanel') ) ? parseInt(getCookie(setting.id + '_lastviewedpanel')) : 0
		this.contentzindex = 1000
		this.playerstate = {autoplay: setting.autoplay, isautoplaying: false, frames: {cur:0, limit:0}, cycles: setting.cycles}
		this.animationstr = ((setting.hingepoint == 'top left')? 'hingetopleft' : 'hingetopright') + ' ' + setting.fxduration + 'ms ease-in-out forwards'
		var slideshow = this
		jQuery(function(){
			slideshow.$container = $('#' + setting.id).css({position: 'relative', overflow: 'visible'})
			if (setting.source != "inline")
				slideshow.getajaxcontent(setting.source)
			else
				slideshow.init()
		})
	}
	
	carddeckslideshow.prototype = {

		navigate:function(action, stopautoplay){

			function stepit(somevar, action){
				switch (action){
					case 'next':
						return (somevar < $contents.length-1)? somevar + 1 : 0
						break
					case 'prev':
						return (somevar > 0)? somevar - 1 : $contents.length-1
						break
					default:
						return Math.min( parseInt(action), $contents.length-1 )
				}
			}
			var slideshow = this
			var setting = this.setting
			var curindex = this.curindex
			var $contents = this.$contents
			var animationstr = this.animationstr
			var contentzindex = this.contentzindex
			var nextindex = stepit(curindex, action)
			this.previndex = curindex
			if (stopautoplay !== false){
				this.pause()
				this.playerstate.isautoplaying = false
			}
			$contents.eq(curindex).css({animation: animationstr, zIndex: ++contentzindex}).addClass(setting.activeclass)
			$contents.eq(nextindex).css({animation: '', zIndex: contentzindex - 1}).removeClass(setting.activeclass)
			if (!transform3d){
				$contents.eq(curindex).css({opacity: 1}).animate({top: '150%', opacity: 0.4}, setting.duration, function(){
					$(this).css({zIndex: -1, top: 0, animation: '', opacity: 1}).removeClass(setting.activeclass)
					setting.onunhinge.call(slideshow, slideshow.curindex, slideshow.previndex)
				})
			}
			this.curindex = nextindex
			this.contentzindex = contentzindex

		}, // end navigate()

		play:function(){ // private function
			var slideshow = this
			this.playerstate.isautoplaying = true
			this.playtimer = setInterval(function(){
				slideshow.navigate('next', false)
				if ( slideshow.playerstate.cycles !=0 && ++slideshow.playerstate.frames.cur >= slideshow.playerstate.frames.limit ){ // if player is in auto playing mode and has reached end of cycle
					slideshow.pause()
					slideshow.playerstate.isautoplaying = false
				}
			}, this.setting.pause)
		},

		pause:function(){ // private function
			clearInterval(this.playtimer)
		},

		userplay:function(){
			clearTimeout(this.playtimer)
			this.playerstate.cycles = 0
			this.play()
		},

		userpause:function(){
			this.playerstate.isautoplaying = false
			this.pause()
		},

	
		init:function(){
			var slideshow = this
			var setting = this.setting
			var $container = this.$container
				.off('mouseenter mouseleave')
				.on('mouseenter', function(){
					slideshow.pause()
				})
				.on('mouseleave', function(){
					if (slideshow.playerstate.isautoplaying){
						slideshow.play()
					}
				})
			var $contents = $container.find('>div.inner')
				.css({transformOrigin: setting.hingepoint, position:'absolute', width:'100%', height:'100%', left:0, top:0})
			this.curindex = Math.min(this.curindex, $contents.length-1)
			this.previndex = this.curindex
			$contents.eq(this.curindex)
				.css({zIndex: ++this.contentzindex})
			$contents.off('animationend').on('animationend', function(){
				$(this).css({zIndex: -1, animation: ''}).removeClass(setting.activeclass)
				setting.onunhinge.call(slideshow, slideshow.curindex, slideshow.previndex)
			})
			this.$contents = $contents
			if (slideshow.playerstate.autoplay && setting.cycles > 0){
				this.playerstate.cycles = setting.cycles
				this.playerstate.frames.cur = 0
				this.playerstate.frames.limit = this.$contents.length * slideshow.playerstate.cycles
			}
			setting.onunhinge.call(slideshow, slideshow.curindex, slideshow.previndex)
			if (slideshow.playerstate.autoplay === true){
				clearInterval(this.playtimer)
				this.play()
			}
			$(window).off('unload.' + setting.id).on('unload.' + setting.id, function(){
				if (setting.persist)
					setCookie(setting.id + '_lastviewedpanel', slideshow.curindex)
				slideshow.$contents.remove()
			})
		}
	}

	return carddeckslideshow

}) (jQuery)
